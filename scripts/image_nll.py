"""
Approximate the bits/dimension for an image model.
"""

import argparse
import os

import numpy as np
import torch.distributed as dist
import wandb

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.logger import wandb_safe_log
from guided_diffusion.script_args import args_to_dict, add_dict_to_argparser
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=True,
    )

    logger.log("evaluating...")
    run_bpd_evaluation(model, diffusion, data, args.num_samples, args.clip_denoised)


def run_bpd_evaluation(
    model,
    diffusion,
    data,
    num_samples,
    clip_denoised,
    step=None,
    task_id=None,
    report_to_wandb=False,
):
    #  TODO: figure out how to do this properly in multi-GPU setup.
    assert dist.get_world_size() == 1, "not implemented!"

    model.eval()
    all_metrics = {
        "vb": [],
        "mse": [],
        "xstart_mse": [],
        "prior_bpd": [],
        "total_bpd": [],
    }
    num_complete = 0
    while num_complete < num_samples:
        batch, model_kwargs = next(data)
        batch = batch.to(dist_util.dev())
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        minibatch_metrics = diffusion.calc_bpd_loop(
            model, batch, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        for key, term_list in all_metrics.items():
            terms = minibatch_metrics[key].sum(dim=0)
            dist.all_reduce(terms)
            term_list.append(terms.detach().cpu().numpy())

        num_complete += dist.get_world_size() * batch.shape[0]

        bpd_so_far = np.sum(all_metrics["total_bpd"]) / num_complete
        logger.log(f"done {num_complete} samples: bpd={bpd_so_far}")

    if dist.get_rank() == 0:
        for name, terms in all_metrics.items():
            out_path = os.path.join(logger.get_dir(), f"{name}_terms.npz")
            logger.log(f"saving {name} terms to {out_path}")
            np.savez(out_path, np.sum(np.stack(terms), axis=0) / num_complete)

        if report_to_wandb:
            metrics = {
                f"validation/{metric}/{task_id}": np.sum(np.stack(values))
                / num_complete
                for metric, values in all_metrics.items()
            }
            wandb_safe_log(metrics, step=step)

    dist.barrier()
    model.train()
    logger.log("evaluation complete")


def create_argparser():
    defaults = dict(
        data_dir="", clip_denoised=True, num_samples=1000, batch_size=1, model_path=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
