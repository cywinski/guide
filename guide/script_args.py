import argparse
import os


def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        learn_sigma=False,
        sigma_small=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )


def classifier_defaults():
    """
    Defaults for classifier models.
    """
    return dict(
        image_size=64,
        classifier_use_fp16=False,
        classifier_width=128,
        classifier_depth=2,
        classifier_attention_resolutions="32,16,8",  # 16
        classifier_use_scale_shift_norm=True,  # False
        classifier_resblock_updown=True,  # False
        classifier_pool="attention",
        # ResNet disjoint classifier parameters
        activetype="ReLU",  # activation type
        pooltype="MaxPool2d",  # Pooling type
        normtype="BatchNorm",  # Batch norm type
        preact=True,  # Places norms and activations before linear/conv layer.
        bn=True,  # Apply Batchnorm.
        affine_bn=True,  # Apply affine transform in BN.
        bn_eps=1e-6,  # Affine transform for batch norm
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        num_channels=128,
        num_res_blocks=3,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        dropout=0.1,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        image_size=32,
        in_channels=3,
        model_switching_timestep=30,
        model_name="UNetModel",
        embedding_kind="concat_time_1hot",  # embedding used for time and "class", possible values in EMBEDDING_KINDS
        model_num_classes=None,
        train_noised_classifier=False,
    )
    res.update(diffusion_defaults())
    return res


def all_training_defaults():
    defaults = dict(
        seed=13,
        data_seed=0,  # Seed used for data generation (mostly train/valid split). Typically, no need to set it.
        wandb_api_key="",
        wandb_experiment_name="test",
        wandb_project_name="project",
        wandb_entity="entity",
        dataroot="data/",
        dataset="CIFAR10",
        schedule_sampler="uniform",
        alpha=4,
        beta=1.2,
        lr=2e-4,
        disjoint_classifier_lr=1e-2,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=64,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=500,
        skip_save=False,
        save_interval=5000,
        guid_generation_interval=1,  # generate new examples from diffusion model every guid_generation_interval steps
        resume_checkpoint="",
        resume_checkpoint_classifier="",
        resume_checkpoint_classifier_noised="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_id=-1,
        reverse=False,
        num_tasks=5,
        limit_tasks=-1,
        limit_classes=-1,
        shared_classes=False,
        train_aug=False,
        skip_normalization=False,
        num_steps=20000,
        scheduler_rate=1.0,
        first_task_num_classes=0,
        first_task_num_steps=-1,  # if -1, set to the same as num_steps.
        skip_gradient_thr=-1,
        log_gradient_stats=False,
        clip_denoised=True,
        cl_method="generative_replay_disjoint_classifier_guidance",  # possible values are defined in CL_METHODS
        use_ddim=False,
        classifier_scale_min_old=None,
        classifier_scale_min_new=None,
        classifier_scale_max_old=None,
        classifier_scale_max_new=None,
        first_task=0,
        gr_n_generated_examples_per_task=32,
        use_old_grad=False,
        use_new_grad=False,
        guid_to_new_classes=False,
        trim_logits=True,
        train_with_disjoint_classifier=False,
        disjoint_classifier_init_num_steps=5000,  # Steps in first task
        disjoint_classifier_num_steps=2000,
        classifier_init_lr=0.1,  # First task learning rate
        classifier_lr=0.05,  # Learning rate
        classifier_weight_decay=5e-4,  # Weight decay
        depth=18,  # ResNet depth
        classifier_augmentation=True,
        diffusion_pretrained_dir=None,  # Directory contating trained diffusion models on each task. It effectively disables any training of the diffusion model.
        negate_old_grad=False,  # negate old gradient
        classifier_first_task_dir=None,
    )
    defaults.update(model_and_diffusion_defaults())
    return defaults


def classifier_and_diffusion_defaults():
    res = classifier_defaults()
    res.update(diffusion_defaults())
    return res


def combine_with_defaults(config):
    res = all_training_defaults()
    for k, v in config.items():
        assert k in res.keys(), "{} not in default values".format(k)
        res[k] = v
    return res


CL_METHODS = [
    "generative_replay",
    "generative_replay_disjoint_classifier_guidance",  # GR with one disjoint classifier
]

EMBEDDING_KINDS = [
    "none",
    "concat_time_1hot",
    "add_time_learned",  # original from the paper
]


def preprocess_args(args):
    """Perform simple validity checks and do a simple initial processing of training args."""

    assert args.cl_method in CL_METHODS
    assert args.embedding_kind in EMBEDDING_KINDS

    if args.first_task_num_steps == -1:
        args.first_task_num_steps = args.num_steps

    if not args.dataroot:
        args.dataroot = os.environ.get("DIFFUSION_DATA", "")

    if (
        args.classifier_scale_min_old is not None
        and args.classifier_scale_max_old is None
    ):
        args.classifier_scale_max_old = args.classifier_scale_min_old

    if (
        args.classifier_scale_min_old is not None
        and args.classifier_scale_max_old is None
    ):
        args.classifier_scale_max_old = args.classifier_scale_min_old

    if (
        args.classifier_scale_max_old is not None
        and args.classifier_scale_min_old is None
    ):
        args.classifier_scale_min_old = args.classifier_scale_max_old

    if (
        args.classifier_scale_min_new is not None
        and args.classifier_scale_max_new is None
    ):
        args.classifier_scale_max_new = args.classifier_scale_min_new

    if (
        args.classifier_scale_min_new is not None
        and args.classifier_scale_max_new is None
    ):
        args.classifier_scale_max_new = args.classifier_scale_min_new

    if (
        args.classifier_scale_max_new is not None
        and args.classifier_scale_min_new is None
    ):
        args.classifier_scale_min_new = args.classifier_scale_max_new


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
