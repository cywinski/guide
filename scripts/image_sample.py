"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.distributed as dist
from torchvision.utils import make_grid

from guide import dist_util, logger
from guide.script_args import (
    add_dict_to_argparser,
    all_training_defaults,
    args_to_dict,
    preprocess_args,
)
from guide.script_util import create_model_and_diffusion, model_and_diffusion_defaults


def main():
    args = create_argparser().parse_args()
    preprocess_args(args)

    os.environ["OPENAI_LOGDIR"] = f"sampled/{args.wandb_experiment_name}"

    dist_util.setup_dist(args)
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0,
                high=args.num_classes,
                size=(args.batch_size,),
                device=dist_util.dev(),
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        all_images.extend(sample.cpu().numpy())
        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        # if args.class_cond:
        #     gathered_labels = [
        #         th.zeros_like(classes) for _ in range(dist.get_world_size())
        #     ]
        #     dist.all_gather(gathered_labels, classes)
        #     all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images)} samples")

    arr = np.concatenate([all_images])
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(
            logger.get_dir(), f"samples_{args.model_path.split('/')[-1][:-3]}.npz"
        )
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    plt.figure()
    plt.axis("off")
    samples_grid = make_grid(th.from_numpy(arr[:16]), 4, normalize=True).permute(
        1, 2, 0
    )
    plt.imshow(samples_grid)
    out_plot = os.path.join(
        logger.get_dir(), f"samples_{args.model_path.split('/')[-1][:-3]}"
    )
    plt.savefig(out_plot)
    logger.log("sampling complete")


def create_argparser():
    defaults = all_training_defaults()
    defaults.update(model_and_diffusion_defaults())
    defaults.update(
        dict(
            model_path="",
            num_samples=2,
            image_size=28,
            in_channels=1,
            model_num_classes=10,
            batch_size=16,
            wandb_experiment_name="test",
        )
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
