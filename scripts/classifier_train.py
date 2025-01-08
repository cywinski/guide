"""
Train a noised image classifier on ImageNet.
"""

import argparse
import copy
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from tqdm import tqdm

import wandb
from dataloaders import base
from dataloaders.datasetGen import *
from dataloaders.utils import prepare_eval_loaders
from guide import dist_util, logger
from guide.convs.ResNetBlock import resnet18
from guide.evaluate import evaluate
from guide.script_args import (
    add_dict_to_argparser,
    all_training_defaults,
    args_to_dict,
)
from guide.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)


def main(args=None, is_sweep=False):
    if args is None:
        args = create_argparser().parse_args()
        if logger.get_rank_without_mpi_import() == 0:
            if args.wandb_api_key:
                os.environ["WANDB_API_KEY"] = args.wandb_api_key
            wandb.init(
                project=args.wandb_project_name,
                name=args.wandb_experiment_name,
                config=args,
                entity=args.wandb_entity,
            )
    os.environ["OPENAI_LOGDIR"] = f"results/classifier/{args.wandb_experiment_name}"
    dist_util.setup_dist(args)
    args.seed = args.seed + logger.get_rank_without_mpi_import()
    seed_everything(args.seed)

    logger.configure()

    logger.log("creating dataset...")
    (
        train_dataset,
        val_dataset,
        image_size,
        image_channels,
        train_transform_classifier,
        rehearsal_transform_classifier,
        train_transform_diffusion,
        n_classes,
        mean_norm,
        std_norm,
    ) = base.__dict__[args.dataset](
        args.dataroot,
        train_aug=False,
        skip_normalization=False,
        standard_norm_stats=args.standard_norm_stats,
        classifier_augmentation=args.classifier_augmentation,
        classifier_training=True,
    )
    args.image_size = image_size
    args.in_channels = image_channels
    args.model_num_classes = n_classes
    args.n_classes = n_classes

    logger.log("creating data loaders...")
    train_dataset_splits, _, classes_per_task = data_split(
        dataset=train_dataset,
        return_classes=True,
        return_task_as_class=False,
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
        return_task_as_class=False,
        num_tasks=args.num_tasks,
        num_classes=n_classes,
        limit_classes=args.limit_classes,
        data_seed=args.data_seed,
        shared_classes=args.shared_classes,
        first_task_num_classes=args.first_task_num_classes,
        validation_frac=0.0,
    )

    val_loaders = prepare_eval_loaders(
        val_dataset_splits, args.num_tasks, args.batch_size
    )

    logger.log("creating classifier model...")
    curr_classifier = resnet18(num_classes=n_classes)
    curr_classifier.to(dist_util.dev())

    curr_classifier = DDP(
        curr_classifier,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    prev_classifier = None
    prev_diffusion_model = None
    prev_diffusion = None
    acc_history = {}
    n_tasks = args.num_tasks if args.limit_tasks == -1 else args.limit_tasks
    global global_step
    global_step = 0
    for task_id in range(n_tasks):
        train_split = train_dataset_splits[task_id]
        val_split = val_dataset_splits[task_id]
        logger.log(f"***** Running training on task {task_id+1}/{args.num_tasks} *****")
        logger.log(f"  Num real examples = {len(train_split)}")

        ## TRAINING ##
        curr_classifier = train_on_task(
            curr_classifier,
            train_split,
            val_split,
            task_id,
            prev_diffusion,
            prev_diffusion_model,
            args,
            prev_classifier,
            rehearsal_transform_classifier,
        )

        ## EVALUATION ##
        accuracies, forgetting, acc_history = evaluate(
            curr_classifier, val_loaders[: task_id + 1], dist_util.dev(), acc_history
        )
        if logger.get_rank_without_mpi_import() == 0:
            for i, (acc, forget) in enumerate(zip(accuracies, forgetting)):
                wandb.log(
                    {f"task_{i}/test/acc": acc, f"task_{i}/test/forgetting": forget},
                    step=global_step,
                )
                logger.log(
                    f"Task {i} - Test accuracy: {acc:.2f} - Forgetting: {forget:.2f}"
                )
            avg_acc = sum(accuracies) / len(accuracies)
            if len(forgetting) == 1:
                avg_forget = 0
            else:
                avg_forget = sum(forgetting) / (len(forgetting) - 1)
            wandb.log(
                {"test/avg_acc": avg_acc, "test/avg_forgetting": avg_forget},
                step=global_step,
            )
            logger.log(f"Average test accuracy: {avg_acc:.2f}")
            logger.log(f"Average test forgetting: {avg_forget:.2f}")
            # Save the model
            if dist.get_rank() == 0:
                model_path = os.path.join(
                    logger.get_dir(), f"classifier_task_{task_id}"
                )
                th.save(curr_classifier.state_dict(), model_path)

        ## LOADING NEXT DIFFUSION MODEL ##
        if task_id < n_tasks - 1:
            logger.log("loading next diffusion model...")
            prev_diffusion_model, prev_diffusion = create_model_and_diffusion(
                **args_to_dict(args, model_and_diffusion_defaults().keys())
            )
            prev_diffusion_model.to(dist_util.dev())
            if args.use_fp16:
                prev_diffusion_model.convert_to_fp16()
            prev_diffusion_model.eval()

            if args.diffusion_dir:
                if dist.get_rank() == 0:
                    prev_diffusion_model_path = find_model_with_highest_step(
                        args.diffusion_dir, task_id
                    )
                    prev_diffusion_model.load_state_dict(
                        dist_util.load_state_dict(
                            prev_diffusion_model_path, map_location=dist_util.dev()
                        )
                    )
                    logger.log(
                        f"Loaded previous diffusion model from {prev_diffusion_model_path}"
                    )
            prev_classifier = copy.deepcopy(curr_classifier)
            prev_classifier.eval()
            prev_classifier.to(dist_util.dev())
            gpu_ok = False
            if th.cuda.is_available():
                device_cap = th.cuda.get_device_capability()
                if device_cap in ((7, 0), (8, 0), (9, 0)):
                    gpu_ok = True

            if not gpu_ok:
                logger.log(
                    "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
                    "than expected."
                )
            else:
                logger.log("Compiling diffusion model")
                prev_diffusion_model = th.compile(
                    prev_diffusion_model, mode="reduce-overhead", fullgraph=True
                )


# Training function
def train_on_task(
    curr_classifier,
    task_dataset,
    val_dataset,
    task_id,
    diffusion,
    diffusion_model,
    args,
    prev_classifier=None,
    rehearsal_transform_classifier=None,
):
    global global_step
    # Initialize the optimizer
    lr = args.lr if task_id != 0 else args.init_lr
    optimizer = th.optim.SGD(
        curr_classifier.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    curr_train_dataloader = th.utils.data.DataLoader(
        task_dataset,
        batch_size=args.batch_size // 2 if task_id > 0 else args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    num_update_steps_per_epoch = len(curr_train_dataloader)
    num_epochs = args.num_epochs
    if task_id == 0:
        num_epochs = args.init_epochs
    num_total_steps = num_update_steps_per_epoch * num_epochs

    # Train!
    prev_generations = None
    prev_class_labels = None

    for epoch in range(num_epochs):
        curr_classifier.train()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=logger.get_rank_without_mpi_import() != 0,
        )
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(curr_train_dataloader):
            real_images = batch[0].to(dist_util.dev())
            real_class_labels_one_hot = batch[1]["y"]
            real_class_labels = th.argmax(real_class_labels_one_hot, dim=1).to(
                dist_util.dev()
            )

            if task_id > 0:
                if prev_generations is None or (
                    step % args.rehearsal_generation_interval == 0
                ):
                    # generate batch of rehearsal samples
                    rehearsal_images, rehearsal_class_labels = sample_examples(
                        n_examples=args.batch_size // 2,
                        batch_size=args.batch_size // 2,
                        diffusion=diffusion,
                        curr_classifier=curr_classifier,
                        prev_classifier=prev_classifier,
                        diffusion_model=diffusion_model,
                        max_class=(args.n_classes // args.num_tasks) * task_id,
                        num_tasks=args.num_tasks,
                        n_classes=args.n_classes,
                        grad_scale=args.grad_scale,
                        use_old_grad=args.use_old_grad,
                        use_new_grad=args.use_new_grad,
                        negate_old_grad=args.negate_old_grad,
                        guid_to_new_classes=args.guid_to_new_classes,
                        use_ddim=args.use_ddim,
                        in_channels=args.in_channels,
                        image_size=args.image_size,
                    )
                    # normalize rehearsal images from [-1, 1] to [0, 1]
                    rehearsal_images = (rehearsal_images + 1) / 2

                    if args.rehearsal_generation_interval > 1:
                        prev_generations = rehearsal_images.cpu()
                        prev_class_labels = rehearsal_class_labels.cpu()
                else:
                    rehearsal_images = prev_generations.to(dist_util.dev())
                    rehearsal_class_labels = prev_class_labels.to(dist_util.dev())
                if args.use_knowledge_distillation:
                    # use soft labels for rehearsal images
                    rehearsal_class_labels = prev_classifier(rehearsal_images)[
                        :, : (args.n_classes // args.num_tasks) * task_id
                    ]
                    rehearsal_class_labels = F.softmax(rehearsal_class_labels, dim=1)
                    rehearsal_class_labels = th.argmax(rehearsal_class_labels, dim=1)
                if rehearsal_transform_classifier is not None:
                    rehearsal_images = rehearsal_transform_classifier(rehearsal_images)

                model_input = th.cat([real_images, rehearsal_images], dim=0)
                class_labels = th.cat(
                    [real_class_labels, rehearsal_class_labels], dim=0
                )
            else:
                model_input = real_images
                class_labels = real_class_labels

            shuffle = th.randperm(model_input.size(0))
            model_input = model_input[shuffle]
            class_labels = class_labels[shuffle]

            model_output = curr_classifier(model_input)
            # train only already seen classes
            model_output = model_output[
                :, : (args.n_classes // args.num_tasks) * (task_id + 1)
            ]
            loss = F.cross_entropy(model_output, class_labels, reduction="none")
            losses = {}
            losses[f"task_{task_id}/train/loss"] = loss.detach()
            losses[f"task_{task_id}/train/lr"] = optimizer.param_groups[0]["lr"]
            losses[f"task_{task_id}/train/acc"] = (
                model_output.argmax(dim=1) == class_labels
            ).sum().item() / class_labels.size(0)
            if logger.get_rank_without_mpi_import() == 0:
                for key, value in losses.items():
                    wandb.log(
                        {key: value},
                        step=global_step,
                    )

            del losses
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.anneal_lr:
                set_annealed_lr(
                    optimizer,
                    lr,
                    (step + epoch * num_update_steps_per_epoch) / num_total_steps,
                )
            if (
                logger.get_rank_without_mpi_import() == 0
                and step % args.save_images_steps == 0
            ):
                for class_id in range((args.n_classes // args.num_tasks) * task_id):
                    # Find indices of images belonging to the current class
                    class_indices = np.where(
                        rehearsal_class_labels.cpu().numpy() == class_id
                    )[0]

                    # Get the selected images
                    if len(class_indices) > 0:
                        selected_images = rehearsal_images[class_indices]

                        # Log the images
                        wandb.log(
                            {
                                f"task_{task_id}/train/class_{class_id}_rehearsal_images": [
                                    wandb.Image(selected_images)
                                ]
                            },
                            step=global_step,
                        )
            progress_bar.update(1)
            global_step += 1
        progress_bar.close()
    return curr_classifier


def sample_examples(
    n_examples,
    batch_size,
    diffusion,
    curr_classifier,
    prev_classifier,
    diffusion_model,
    max_class,
    num_tasks,
    n_classes,
    grad_scale=1.0,
    use_old_grad=True,
    use_new_grad=True,
    negate_old_grad=False,
    guid_to_new_classes=True,
    use_ddim=False,
    in_channels=3,
    image_size=32,
):
    all_images = []
    all_labels = []

    # NOTE: Possible further improvements from http://arxiv.org/abs/2302.07121, but with them
    # sampling becomes very time-consuming.
    def cond_fn(x, t, pred_xstart, y=None):
        assert y is not None
        with th.enable_grad():
            if not use_old_grad and not use_new_grad:
                return th.zeros_like(x)
            if use_old_grad and use_new_grad:
                raise NotImplementedError()

            x_in = pred_xstart
            x_out = x

            if use_old_grad:
                logits_old = prev_classifier(x_in)
                logits_old = logits_old[:, :max_class]
                loss = -F.cross_entropy(logits_old, y, reduction="none")
                if negate_old_grad:
                    loss = -loss
            elif use_new_grad:
                logits_new = curr_classifier(x_in)
                logits_new = logits_new[:, : max_class + (n_classes // num_tasks)]

                if guid_to_new_classes:
                    probs_new = F.softmax(logits_new[:, max_class:], dim=-1)
                    most_probable_new_class = th.argmax(probs_new, dim=-1) + max_class
                    loss = -F.cross_entropy(
                        logits_new, most_probable_new_class, reduction="none"
                    )
                else:
                    loss = -F.cross_entropy(logits_new, y, reduction="none")
            else:
                return th.zeros_like(x_out)

            grad = th.autograd.grad(loss.sum(), x_out)[0]
            return grad * grad_scale

    diffusion_model.eval()
    while len(all_images) * batch_size < n_examples:
        model_kwargs = {}
        classes = (
            th.randint(0, max_class, size=(batch_size,)).long().to(dist_util.dev())
        )

        model_kwargs["y"] = classes
        sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop
        sample = sample_fn(
            diffusion_model,
            (
                len(classes),
                in_channels,
                image_size,
                image_size,
            ),
            clip_denoised=True,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn if use_old_grad or use_new_grad else None,
            device=dist_util.dev(),
            compute_grads=use_old_grad or use_new_grad,
        )

        sample = sample.detach()
        all_images.extend([sample])
        all_labels.extend([classes])
        th.cuda.empty_cache()
    all_images = th.cat(all_images)
    all_labels = th.cat(all_labels)
    return all_images, all_labels


def find_model_with_highest_step(model_dir, task_id):
    """Find the model file with the highest step number for a given task ID.

    Args:
        model_dir: Directory containing model files
        task_id: Task ID to find model for

    Returns:
        Path to model file with highest step number
    """
    import os
    import re

    # Find all model files for this task ID
    model_files = []
    pattern = f"ema_.*_{task_id}.pt"
    for f in os.listdir(model_dir):
        if re.match(pattern, f):
            model_files.append(f)

    if not model_files:
        raise ValueError(f"No model files found for task {task_id} in {model_dir}")

    # Extract step numbers using regex
    step_pattern = re.compile(r"ema_[\d\.]+_(\d+)_")
    steps = []
    for f in model_files:
        match = step_pattern.search(f)
        if match:
            steps.append((int(match.group(1)), f))

    if not steps:
        raise ValueError(
            f"No valid step numbers found in model filenames for task {task_id}"
        )

    # Return path to model with highest step
    highest_step_file = max(steps, key=lambda x: x[0])[1]
    return os.path.join(model_dir, highest_step_file)


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


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
    defaults = dict(
        batch_size=4,
        microbatch=-1,
        standard_norm_stats=False,
        classifier_augmentation=False,
        lr=0.01,
        init_lr=0.01,
        weight_decay=5e-5,
        momentum=0.0,
        num_workers=4,
        init_epochs=1,
        num_epochs=1,
        rehearsal_generation_interval=1,
        grad_scale=1.0,
        use_old_grad=False,
        use_new_grad=False,
        negate_old_grad=False,
        guid_to_new_classes=True,
        anneal_lr=False,
        save_images_steps=-1,
        diffusion_dir="",
        seed=1,
        limit_tasks=-1,
        use_knowledge_distillation=False,
    )
    defaults.update(all_training_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
