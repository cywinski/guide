import inspect

import matplotlib
import numpy as np

from . import gaussian_diffusion as gd
from .logger import wandb_safe_log
from .resnet import ResNet
from .respace import SpacedDiffusion, space_timesteps
from .script_args import model_and_diffusion_defaults
from .unet import EncoderUNetModel, SuperResModel

# NUM_CLASSES = 1000


def create_model_and_diffusion(
    image_size,
    in_channels,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    model_name,
    model_switching_timestep,
    embedding_kind,
    model_num_classes=None,
    noise_marg_reg=False,
    train_noised_classifier=False,
    classifier_augmentation=True,
):
    model = create_model(
        image_size,
        in_channels,
        num_channels,
        num_res_blocks,
        model_name=model_name,
        model_switching_timestep=model_switching_timestep,
        embedding_kind=embedding_kind,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        num_classes=model_num_classes,
        classifier_augmentation=classifier_augmentation,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        noise_marg_reg=noise_marg_reg,
        train_noised_classifier=train_noised_classifier,
    )

    return model, diffusion


def create_model(
    image_size,
    in_channels,
    num_channels,
    num_res_blocks,
    model_name,
    model_switching_timestep,
    embedding_kind,
    channel_mult="",
    learn_sigma=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    num_classes=None,
    classifier_augmentation=False,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        elif image_size == 28:
            channel_mult = (1, 2, 2)
        elif image_size == 1:
            channel_mult = (1, 1, 1)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if model_name == "UNetModel":
        print("Using single model")
        from .unet import UNetModel as Model
    elif model_name == "MLPModel":
        from .mlp import MLPModel as Model
    else:
        raise NotImplementedError
    return Model(
        image_size=image_size,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=(in_channels if not learn_sigma else in_channels * 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        model_switching_timestep=model_switching_timestep,
        embedding_kind=embedding_kind,
        classifier_augmentation=classifier_augmentation,
    )


def create_classifier_and_diffusion(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
    learn_sigma,
    sigma_small,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
):
    classifier = create_classifier(
        image_size,
        classifier_use_fp16,
        classifier_width,
        classifier_depth,
        classifier_attention_resolutions,
        classifier_use_scale_shift_norm,
        classifier_resblock_updown,
        classifier_pool,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return classifier, diffusion


def create_classifier(
    image_size,
    classifier_use_fp16,
    classifier_width,
    classifier_depth,
    classifier_attention_resolutions,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_pool,
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return EncoderUNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=classifier_width,
        out_channels=1000,
        num_res_blocks=classifier_depth,
        attention_resolutions=tuple(attention_ds),
        channel_mult=channel_mult,
        use_fp16=classifier_use_fp16,
        num_head_channels=64,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
    )


def create_resnet_classifier(args):
    classifier = ResNet(args)
    return classifier


def sr_model_and_diffusion_defaults():
    res = model_and_diffusion_defaults()
    res["large_size"] = 256
    res["small_size"] = 64
    arg_names = inspect.getfullargspec(sr_create_model_and_diffusion)[0]
    for k in res.copy().keys():
        if k not in arg_names:
            del res[k]
    return res


def sr_create_model_and_diffusion(
    large_size,
    small_size,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
):
    model = sr_create_model(
        large_size,
        small_size,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def sr_create_model(
    large_size,
    small_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
):
    _ = small_size  # hack to prevent unused variable

    if large_size == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif large_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {large_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(large_size // int(res))

    return SuperResModel(
        image_size=large_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    predict_xprevious=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    noise_marg_reg=False,
    train_noised_classifier=False,
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]

    if predict_xstart:
        model_mean_type = gd.ModelMeanType.START_X
    elif predict_xprevious:
        model_mean_type = gd.ModelMeanType.PREVIOUS_X
    else:
        model_mean_type = gd.ModelMeanType.EPSILON

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        noise_marg_reg=noise_marg_reg,
        train_noised_classifier=train_noised_classifier,
    )


def dict2array(results):
    tasks = len(results[0])
    array = np.zeros((tasks, tasks))
    for e, (key, val) in enumerate(reversed(results.items())):
        for e1, (k, v) in enumerate(reversed(val.items())):
            array[tasks - int(e1) - 1, tasks - int(e) - 1] = round(v, 3)
    return np.transpose(array, axes=(1, 0))


def grid_plot(ax, array, type):
    if type == "fid":
        round = 1
    else:
        round = 2
    avg_array = np.around(array, round)
    num_tasks = array.shape[1]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["#287233", "#4c1c24"]
    )
    ax.imshow(avg_array, vmin=50, vmax=300, cmap=cmap)
    for i in range(len(avg_array)):
        for j in range(avg_array.shape[1]):
            if j >= i:
                ax.text(
                    j,
                    i,
                    avg_array[i, j],
                    va="center",
                    ha="center",
                    c="w",
                    fontsize=70 / num_tasks,
                )
    ax.set_yticks(np.arange(num_tasks))
    ax.set_ylabel("Number of tasks")
    ax.set_xticks(np.arange(num_tasks))
    ax.set_xlabel("Tasks finished")
    ax.set_title(
        f"{type} -- {np.round(np.mean(array[:, -1]), 3)} -- std {np.round(np.std(array[:, -1]), 2)}"
    )


def results_to_log(
    test_acc_table,
    train_acc_table,
    validation_time,
    step,
    task_id,
):
    log_dict = {"validation_time": validation_time}
    avg_acc_train = 0.0
    avg_acc_test = 0.0
    avg_forgetting_train = 0.0
    avg_forgetting_test = 0.0
    for j in range(task_id + 1):
        log_dict.update(
            {
                f"test/accuracy/{j}": test_acc_table[j][task_id],
                f"train/accuracy/{j}": train_acc_table[j][task_id],
            }
        )
        avg_acc_train += train_acc_table[j][task_id]
        avg_acc_test += test_acc_table[j][task_id]

        max_forgetting_train = 0.0 if (j == task_id) else -float("inf")
        max_forgetting_test = 0.0 if (j == task_id) else -float("inf")
        for k in range(j, task_id):
            max_forgetting_train = max(
                max_forgetting_train,
                train_acc_table[j][k] - train_acc_table[j][task_id],
            )
            max_forgetting_test = max(
                max_forgetting_test, test_acc_table[j][k] - test_acc_table[j][task_id]
            )
        avg_forgetting_train += max_forgetting_train
        avg_forgetting_test += max_forgetting_test

    log_dict.update(
        {
            "test/avg_accuracy": avg_acc_test / (task_id + 1),
            "train/avg_accuracy": avg_acc_train / (task_id + 1),
            "train/avg_forgetting": (
                (avg_forgetting_train / task_id) if task_id > 0 else 0.0
            ),
            "test/avg_forgetting": (
                (avg_forgetting_test / task_id) if task_id > 0 else 0.0
            ),
        }
    )

    wandb_safe_log(log_dict, step=step)

    # return PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
