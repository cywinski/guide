"""
Train a diffusion model on images.
"""

import sys

sys.path.append(".")
import argparse
import copy
import os
import time
from collections import OrderedDict

import numpy as np
import torch as th
import wandb

from cl_methods.utils import get_cl_method
from dataloaders import base
from dataloaders.datasetGen import *
from dataloaders.utils import prepare_eval_loaders
from evaluations.validation import Validator
from guided_diffusion import dist_util, logger
from guided_diffusion.logger import wandb_safe_log
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_args import (
    add_dict_to_argparser,
    all_training_defaults,
    args_to_dict,
    classifier_defaults,
    preprocess_args,
)
from guided_diffusion.script_util import (
    create_classifier,
    create_model_and_diffusion,
    create_resnet_classifier,
    model_and_diffusion_defaults,
    results_to_log,
)
from guided_diffusion.train_util import TrainLoop

# os.environ["WANDB_MODE"] = "disabled"


def main():
    args = create_argparser().parse_args()
    run_training_with_args(args)


def run_training_with_args(args):
    preprocess_args(args)

    dist_util.setup_dist(args)

    if logger.get_rank_without_mpi_import() == 0:
        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key

        # On one of the clusters I use, wandb init sometimes randomly fails because of some networking issues.
        # Retry several times.
        for _ in range(10):
            try:
                wandb.init(
                    project="diffusion_guidance_cl",
                    name=args.experiment_name,
                    config=args,
                    entity="cl-diffusion",
                )
            except:
                time.sleep(5)
            else:
                break

    random_generator = None
    print("Using manual seed = {}".format(args.seed))
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    th.cuda.manual_seed(args.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    random_generator = th.Generator()
    random_generator.manual_seed(args.seed)
    os.environ["OPENAI_LOGDIR"] = f"results/{args.experiment_name}"
    os.makedirs(os.path.join(logger.get_dir(), "generated_examples"), exist_ok=True)
    logger.configure()

    (
        train_dataset,
        val_dataset,
        image_size,
        image_channels,
        train_transform_classifier,
        train_transform_diffusion,
        n_classes,
    ) = base.__dict__[args.dataset](
        args.dataroot,
        train_aug=args.train_aug,
        skip_normalization=args.skip_normalization,
        classifier_augmentation=args.classifier_augmentation,
    )

    args.image_size = image_size
    args.in_channels = image_channels
    args.model_num_classes = n_classes

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.log_gradient_stats and not os.environ.get("WANDB_MODE") == "disabled":
        wandb.watch(model, log_freq=10)
    # if we are not training diffusion, we will not need this model
    if args.class_cond:
        model.to(dist_util.dev())
    print(model)

    classifier = None
    if args.train_with_disjoint_classifier:
        print("creating disjoint classifier...")
        if args.classifier_type == "resnet":
            defaults = classifier_defaults()
            parser = argparse.ArgumentParser()
            add_dict_to_argparser(parser, defaults)
            opts = parser.parse_args([])
            opts.model_num_classes = n_classes
            opts.in_channels = args.in_channels
            opts.classifier_augmentation = args.classifier_augmentation
            opts.depth = args.depth
            opts.noised = args.train_noised_classifier
            classifier = create_resnet_classifier(opts)
        elif args.classifier_type == "unet":
            classifier = create_classifier(
                args.image_size,
                classifier_use_fp16=False,
                classifier_width=128,
                classifier_depth=2,
                classifier_attention_resolutions="32,16,8",  # 16
                classifier_use_scale_shift_norm=True,  # False
                classifier_resblock_updown=True,  # False
                classifier_pool="attention",
            )
        classifier.to(dist_util.dev())
        # Needed for creating correct EMAs and fp16 parameters.
        dist_util.sync_params(classifier.parameters())
        print(classifier)

        if args.resume_checkpoint_classifier:
            classifier.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint_classifier, map_location=dist_util.dev()
                )
            )
            print(f"loading classifier from {args.resume_checkpoint_classifier}...")

    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion, args
    )

    logger.log("creating data loader...")

    train_dataset_splits, _, classes_per_task = data_split(
        dataset=train_dataset,
        return_classes=True,
        return_task_as_class=args.use_task_index,
        num_tasks=args.num_tasks,
        num_classes=n_classes,
        limit_classes=args.limit_classes,
        data_seed=args.data_seed,
        shared_classes=args.shared_classes,
        first_task_num_classes=args.first_task_num_classes,
        validation_frac=0.0,
    )

    val_dataset_splits, _, classes_per_task = data_split(
        dataset=val_dataset,
        return_classes=True,
        return_task_as_class=args.use_task_index,
        num_tasks=args.num_tasks,
        num_classes=n_classes,
        limit_classes=args.limit_classes,
        data_seed=args.data_seed,
        shared_classes=args.shared_classes,
        first_task_num_classes=args.first_task_num_classes,
        validation_frac=0.0,
    )

    train_loaders = []
    validation_loaders = None
    if not args.skip_validation:
        validation_loaders = prepare_eval_loaders(
            train_dataset_splits=train_dataset_splits,
            val_dataset_splits=val_dataset_splits,
            args=args,
            include_train=False,
            generator=random_generator,
        )
        fid_eval_loaders = prepare_eval_loaders(
            train_dataset_splits=train_dataset_splits,
            val_dataset_splits=val_dataset_splits,
            args=args,
            include_train=args.eval_fid_on_train_and_valid,
            generator=random_generator,
        )

        #  Bump the version below if we want to invalidate all previous caches.
        stats_file_name = (
            f"v2_data_seed_{args.data_seed}_num_tasks_{args.num_tasks}_"
            f"limit_classes_{args.limit_classes}_use_train_and_valid_{args.eval_fid_on_train_and_valid}"
        )
        if args.use_gpu_for_validation:
            device_for_validation = dist_util.dev()
        else:
            device_for_validation = th.device("cpu")
        if args.dataset.lower() != "cern":
            validator = Validator(
                n_classes=n_classes,
                device=dist_util.dev(),
                dataset=args.dataset,
                stats_file_name=stats_file_name,
                score_model_device=device_for_validation,
                fid_dataloaders=fid_eval_loaders,
                clf_dataloaders=validation_loaders,
                force_inception_for_fid=args.force_inception_for_fid,
            )
        else:
            raise NotImplementedError()  # Adapt CERN validator
            # validator = CERN_Validator(dataloaders=val_loaders, stats_file_name=stats_file_name, device=dist_util.dev())

    test_acc_table = OrderedDict()
    train_acc_table = OrderedDict()

    train_loop = None
    cl_method = get_cl_method(args)
    global_step = 0
    dataset_yielder = None
    train_loader = None
    generated_previous_examples = None

    if args.limit_tasks != -1:
        n_tasks = args.limit_tasks
    else:
        n_tasks = args.num_tasks

    for task_id in range(n_tasks):
        if args.first_task_num_classes > 0 and task_id == 0:
            max_class = args.first_task_num_classes
        else:
            max_class = (
                ((task_id + 1) * (n_classes // n_tasks)) - 1
            ) + args.first_task_num_classes

        if task_id == 0:
            if args.class_cond:
                num_steps = args.first_task_num_steps
            else:
                num_steps = args.disjoint_classifier_init_num_steps
        else:
            if args.class_cond:
                num_steps = args.num_steps
            else:
                num_steps = args.disjoint_classifier_num_steps

        logger.log("moving real dataset to gpu...")
        train_dataset_splits[task_id].dataset.dataset.dataset = train_dataset_splits[
            task_id
        ].dataset.dataset.dataset.to(dist_util.dev())
        train_dataset_splits[task_id].dataset.dataset.labels = train_dataset_splits[
            task_id
        ].dataset.dataset.labels.to(dist_util.dev())

        train_loop = TrainLoop(
            params=args,
            model=model,
            prev_model=copy.deepcopy(model).to(dist_util.dev()),
            diffusion=diffusion,
            task_id=task_id,
            data=train_dataset_splits[task_id],
            data_yielder=None,
            data_loader=None,
            batch_size=args.batch_size,
            microbatch=args.microbatch,
            lr=args.lr,
            scheduler_rate=args.scheduler_rate,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            skip_save=args.skip_save,
            save_interval=args.save_interval,
            plot_interval=args.plot_interval,
            resume_checkpoint=(
                args.resume_checkpoint if task_id == args.first_task else None
            ),
            use_fp16=args.use_fp16,
            fp16_scale_growth=args.fp16_scale_growth,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.lr_anneal_steps,
            num_steps=num_steps,
            image_size=args.image_size,
            in_channels=args.in_channels,
            class_cond=args.class_cond,
            max_class=max_class,
            global_steps_before=global_step,
            cl_method=cl_method,
            validation_loaders=validation_loaders,
            use_task_index=args.use_task_index,
            scale_classes_loss=args.scale_classes_loss,
            classes_per_task=classes_per_task,
            use_ddim=args.use_ddim,
            classifier_scale_min_old=args.classifier_scale_min_old,
            classifier_scale_min_new=args.classifier_scale_min_new,
            classifier_scale_max_old=args.classifier_scale_max_old,
            classifier_scale_max_new=args.classifier_scale_max_new,
            guid_generation_interval=args.guid_generation_interval,
            use_old_grad=args.use_old_grad,
            use_new_grad=args.use_new_grad,
            guid_to_new_classes=args.guid_to_new_classes,
            trim_logits=args.trim_logits,
            disjoint_classifier=classifier,
            prev_disjoint_classifier=copy.deepcopy(classifier),
            diffusion_pretrained_dir=args.diffusion_pretrained_dir,
            train_transform_classifier=train_transform_classifier,
            train_transform_diffusion=train_transform_diffusion,
            norm_grads=args.norm_grads,
            n_classes=n_classes,
            random_generator=random_generator,
            classifier_first_task_dir=args.classifier_first_task_dir,
            train_noised_classifier=args.train_noised_classifier,
        )

        if task_id >= args.first_task:
            (
                dataset_yielder,
                train_loader,
                generated_previous_examples,
            ) = cl_method.get_data_for_task(
                dataset=train_dataset_splits[task_id],
                task_id=task_id,
                train_loop=train_loop,
                generator=random_generator,
                step=global_step,
            )
            train_loaders.append(train_loader)
            logger.log(f"training task {task_id}")

            train_loop.data_yielder = dataset_yielder
            train_loop.data_loader = train_loader

        train_loop_start_time = time.time()
        if task_id >= args.first_task:
            train_loop.run_loop()
        global_step += num_steps
        train_loop_time = time.time() - train_loop_start_time
        wandb_safe_log({"train_loop_time": train_loop_time}, step=global_step)

        test_acc_table[task_id] = OrderedDict()
        train_acc_table[task_id] = OrderedDict()
        validation_start_time = time.time()
        for j in range(task_id + 1):
            if args.train_with_disjoint_classifier:
                clf_results = validator.calculate_accuracy_with_classifier(
                    model=(
                        model if not args.train_with_disjoint_classifier else classifier
                    ),
                    task_id=j,
                    train_loader=(
                        train_loaders[j - args.first_task]
                        if j >= args.first_task
                        else None
                    ),
                    max_class=max_class,
                    train_with_disjoint_classifier=args.train_with_disjoint_classifier,
                )
                test_acc_table[j][task_id] = clf_results["accuracy"]["test"]
                train_acc_table[j][task_id] = clf_results["accuracy"]["train"]
                logger.log(f"Test accuracy task {j}: {test_acc_table[j][task_id]}")
                logger.log(f"Train accuracy task {j}: {train_acc_table[j][task_id]}")
            else:
                test_acc_table[j][task_id] = 0.0
                train_acc_table[j][task_id] = 0.0

        validation_time = time.time() - validation_start_time
        if logger.get_rank_without_mpi_import() == 0:
            results_to_log(
                test_acc_table,
                train_acc_table,
                validation_time=validation_time,
                step=global_step,
                task_id=task_id,
            )
        if generated_previous_examples is not None:
            th.save(
                generated_previous_examples,
                os.path.join(
                    logger.get_dir(), f"generated_examples/task_{task_id:02d}.pt"
                ),
            )
        train_loop.prev_ddp_model = copy.deepcopy(model)
    print("TEST ACCURACY TABLE:")
    print(test_acc_table)
    print("TRAIN ACCURACY TABLE:")
    print(train_acc_table)


def create_argparser():
    defaults = all_training_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
