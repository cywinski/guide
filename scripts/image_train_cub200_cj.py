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
import torchvision
import numpy as np
import torch as th

import wandb
from cl_methods.utils import get_cl_method
from dataloaders import base
from dataloaders.datasetGen import *
from dataloaders.utils import prepare_eval_loaders
from guide import dist_util, logger
from guide.logger import wandb_safe_log
from guide.resample import create_named_schedule_sampler
from guide.script_args import (
    add_dict_to_argparser,
    all_training_defaults,
    args_to_dict,
    classifier_defaults,
    preprocess_args,
)
from guide.script_util import (
    create_model_and_diffusion,
    create_resnet_classifier,
    model_and_diffusion_defaults,
    results_to_log,
)
from dataloaders.utils import yielder

from guide.train_util_cub200_cj import TrainLoop
import torch.distributed as dist

# os.environ["WANDB_MODE"] = "disabled"


def main():
    args = create_argparser().parse_args()
    run_training_with_args(args)

def calculate_accuracy(model, validation_loader, device='cuda' if th.cuda.is_available() else 'cpu', is_cub=False):
    model.eval()  # Set the model to evaluation mode
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with th.no_grad():  # Disable gradient calculation
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            if is_cub:
                extended_labels = th.cat((th.zeros(labels.size(0), 1000, device=device), labels), dim=1)
            else:
                extended_labels = th.cat((labels, th.zeros(labels.size(0), 200, device=device)), dim=1)


            # Convert one-hot to class indices
            _, true_labels = th.max(extended_labels, 1)

            # Top-1 accuracy
            _, predicted_top1 = th.max(outputs.data, 1)
            correct_top1 += (predicted_top1 == true_labels).sum().item()

            # Top-5 accuracy
            _, predicted_top5 = outputs.topk(5, 1, largest=True, sorted=True)
            correct_top5 += (predicted_top5 == true_labels.view(-1, 1)).sum().item()
            total += labels.size(0)

    accuracy_top1 = 100 * correct_top1 / total
    accuracy_top5 = 100 * correct_top5 / total
    return accuracy_top1, accuracy_top5

def run_training_with_args(args):
    preprocess_args(args)

    dist_util.setup_dist(args)

    if logger.get_rank_without_mpi_import() == 0:
        if args.wandb_api_key:
            os.environ["WANDB_API_KEY"] = args.wandb_api_key
        wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_experiment_name,
            config=args,
            entity=args.wandb_entity,
        )

    args.seed = args.seed + logger.get_rank_without_mpi_import()
    random_generator = seed_everything(args.seed)
    os.environ["OPENAI_LOGDIR"] = f"results/{args.wandb_experiment_name}"
    os.makedirs(os.path.join(logger.get_dir(), "generated_examples"), exist_ok=True)
    logger.configure()
    logger.log("Using manual seed = {}".format(args.seed))

    (
        train_dataset_cub,
        val_dataset_cub,
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

    (
        train_dataset_imagenet,
        val_dataset_imagenet,
        _,
        _,
        _,
        _,
        _,
    ) = base.__dict__[args.dataset](
        args.dataroot,
        train_aug=args.train_aug,
        skip_normalization=args.skip_normalization,
        classifier_augmentation=args.classifier_augmentation,
    )

    args.image_size = image_size
    args.in_channels = image_channels
    args.model_num_classes = 1000

    logger.log("Loading pretrained ResNet18 model...")
    classifier = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    # Get the number of features in the last layer
    num_ftrs = classifier.fc.in_features

    # Store the original weights and bias
    original_weights = classifier.fc.weight.data
    original_bias = classifier.fc.bias.data

    new_fc = th.nn.Linear(num_ftrs, 1200)

    # Copy the weights and bias for the original 1000 classes
    new_fc.weight.data[:1000] = original_weights
    new_fc.bias.data[:1000] = original_bias

    # Initialize the weights for the new 200 classes
    th.nn.init.xavier_uniform_(new_fc.weight.data[1000:])
    new_fc.bias.data[1000:].fill_(0.01)

    # Replace the old fc layer with the new one
    classifier.fc = new_fc
    classifier.to(dist_util.dev())
    dist_util.sync_params(classifier.parameters())

    val_loader_cub = th.utils.data.DataLoader(
            dataset=val_dataset_cub,
            batch_size=args.batch_size,
            shuffle=False,
            generator=random_generator,
        )

    val_loader_imagenet = th.utils.data.DataLoader(
            dataset=val_dataset_imagenet,
            batch_size=args.batch_size,
            shuffle=False,
            generator=random_generator,
        )

    train_loader = th.utils.data.DataLoader(
            dataset=train_dataset_cub,
            batch_size=args.batch_size // 2,
            shuffle=True,
            drop_last=True,
            generator=random_generator,
        )
    dataset_yielder = yielder(train_loader)

    train_loader_imagenet = th.utils.data.DataLoader(
            dataset=train_dataset_imagenet,
            batch_size=args.batch_size // 2,
            shuffle=True,
            drop_last=True,
            generator=random_generator,
        )
    dataset_yielder_imagenet = yielder(train_loader_imagenet)

    train_loop = None
    cl_method = get_cl_method(args)
    global_step = 0
    num_steps = args.disjoint_classifier_init_num_steps

    train_loop = TrainLoop(
        params=args,
        model=None,
        prev_model=None,
        diffusion=None,
        task_id=1,
        data=train_dataset_cub,
        data_yielder=dataset_yielder,
        data_loader=train_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        scheduler_rate=args.scheduler_rate,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        skip_save=args.skip_save,
        save_interval=args.save_interval,
        resume_checkpoint=(
            args.resume_checkpoint
        ),
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=None,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        num_steps=num_steps,
        image_size=args.image_size,
        in_channels=args.in_channels,
        max_class=1199,
        global_steps_before=global_step,
        cl_method=cl_method,
        classes_per_task=200,
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
        n_classes=1200,
        random_generator=random_generator,
        classifier_first_task_dir=args.classifier_first_task_dir,
        train_noised_classifier=args.train_noised_classifier,
        val_loader_imagenet=val_loader_imagenet,
        val_loader_cub=val_loader_cub,
        train_yielder_imagenet=dataset_yielder_imagenet,
    )

    train_loop_start_time = time.time()
    train_loop.run_loop()
    global_step += num_steps
    train_loop_time = time.time() - train_loop_start_time
    wandb_safe_log({"train_loop_time": train_loop_time}, step=global_step)

    logger.log("validation...")
    validation_start_time = time.time()
    val_accuracy_imagenet_top1, val_accuracy_imagenet_top5 = calculate_accuracy(classifier, val_loader_imagenet)
    val_accuracy_cub_top1, val_accuracy_cub_top5 = calculate_accuracy(classifier, val_loader_cub, is_cub=True)
    validation_time = time.time() - validation_start_time
    if logger.get_rank_without_mpi_import() == 0:
                            wandb_safe_log({"test/accuracy_imagenet@1": val_accuracy_imagenet_top1, "test/accuracy_cub200@1": val_accuracy_cub_top1,}, step=global_step)
                            logger.log(f"Validation accuracy@1 on ImageNet final: {val_accuracy_imagenet_top1}")
                            logger.log(f"Validation accuracy@1 on CUB-200 final: {val_accuracy_cub_top1}")


def seed_everything(seed):
    th.manual_seed(seed)
    np.random.seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    random_generator = th.Generator()
    random_generator.manual_seed(seed)
    return random_generator


def create_argparser():
    defaults = all_training_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
