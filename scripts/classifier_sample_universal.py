"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.

python scripts/classifier_sample_universal.py --num_samples=100 --use_ddim=True --timestep_respacing=ddim25 --model_path=results_new/bc_cifar100_ci5_class_cond_diffusion_long/ema_0.9999_100000_0.pt --classifier_path=models/resnet18_cifar100_task0.pt --num_classes_sample=20 --model_num_classes=100 --batch_size=50
"""

import argparse
import os

import numpy as np
import torch as th
import torch.nn.functional as F

from guide import dist_util, logger
from guide.script_args import add_dict_to_argparser, args_to_dict, classifier_defaults
from guide.script_util import (
    create_model_and_diffusion,
    create_resnet_classifier,
    model_and_diffusion_defaults,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    args = create_argparser().parse_args()
    print("Using manual seed = {}".format(args.seed))
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    os.environ["OPENAI_LOGDIR"] = f"out/samples/{args.wandb_experiment_name}"

    assert args.num_samples % args.batch_size == 0

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

    logger.log("loading classifier...")
    defaults = classifier_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    opts = parser.parse_args([])
    opts.model_num_classes = int(args.model_num_classes)
    opts.in_channels = 3
    opts.depth = 18
    opts.noised = False
    classifier = create_resnet_classifier(opts)
    if args.classifier_path:
        classifier.load_state_dict(th.load(args.classifier_path, map_location="cpu"))
    classifier.to(dist_util.dev())
    print(classifier)

    # NOTE: Possible further improvements from http://arxiv.org/abs/2302.07121, but with them
    # sampling becomes very time-consuming.
    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x = x.detach().requires_grad_(True)
            my_t = th.tensor(
                [diffusion.timestep_map.index(ts) for ts in t], device=dist_util.dev()
            )
            out = diffusion.p_mean_variance(
                model, x, my_t, clip_denoised=True, model_kwargs=model_kwargs
            )
            x_in = out["pred_xstart"]

            logit = classifier(x_in)
            if args.trim_logits:
                logit = logit[:, : int(args.num_classes_sample)]

            loss = -F.cross_entropy(logit, y, reduction="none")

            grad = th.autograd.grad(loss.sum(), x)[0]
            return grad * classfier_scale_vec.view(-1, 1, 1, 1)

    def model_fn(x, t, y=None):
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    batch_num = 0
    while len(all_images) < args.num_samples:
        model_kwargs = {}
        model_kwargs["y"] = th.randint(
            args.min_class_sample, args.max_class_sample, (args.batch_size,)
        ).to(dist_util.dev())

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        classfier_scale_vec = (
            th.from_numpy(
                np.random.uniform(
                    low=args.classifier_scale_min,
                    high=args.classifier_scale_max,
                    size=(len(model_kwargs["y"]),),
                )
            )
            .float()
            .to(dist_util.dev())
        )
        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels, args.image_size, args.image_size),
            clip_denoised=True,
            model_kwargs=model_kwargs,
            cond_fn=(
                None
                if args.classifier_scale_min == 0.0 and args.classifier_scale_max == 0.0
                else cond_fn
            ),
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_images.extend([sample.cpu().numpy() for sample in sample])
        all_labels.extend([labels.cpu().numpy() for labels in model_kwargs["y"]])
        logger.log(f"created {len(all_images)} samples")

        batch_num += 1

    arr = all_images
    arr = arr[: args.num_samples]
    label_arr = all_labels
    label_arr = label_arr[: args.num_samples]
    out_path = os.path.join(
        os.path.dirname(args.model_path), f"{args.wandb_experiment_name}.npz"
    )
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr, label_arr)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        timestep_respacing="",
        model_path="",
        classifier_path="",
        classifier_scale_min=0.0,
        classifier_scale_max=0.0,
        wandb_experiment_name="test",
        model_num_classes=10,
        trim_logits=True,
        min_class_sample=0,
        max_class_sample=0,
        seed=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
