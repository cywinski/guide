import copy
import functools
import os
import time

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
from tqdm import tqdm

from dataloaders.utils import yielder
from dataloaders.wrapper import AppendName

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .logger import wandb_safe_log
from .nn import update_ema
from .resample import LossAwareSampler, TaskAwareSampler, UniformSampler
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
# INITIAL_LOG_LOSS_SCALE = 20.0
from .script_util import create_gaussian_diffusion


class TrainLoop:
    def __init__(
        self,
        *,
        params,
        model,
        prev_model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        skip_save,
        save_interval,
        plot_interval,
        resume_checkpoint,
        task_id,
        cl_method,
        classes_per_task,
        train_transform_classifier,
        train_transform_diffusion,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        scheduler_rate=1,
        scheduler_step=1000,
        num_steps=10000,
        image_size=32,
        in_channels=3,
        class_cond=False,
        use_task_index=False,
        max_class=None,
        global_steps_before=0,
        validation_loaders=None,
        scale_classes_loss=False,
        use_ddim=False,
        classifier_scale_min_old=None,
        classifier_scale_min_new=None,
        classifier_scale_max_old=None,
        classifier_scale_max_new=None,
        guid_generation_interval=1000,
        use_old_grad=True,
        use_new_grad=True,
        guid_to_new_classes=False,
        trim_logits=False,
        data_yielder=None,
        data_loader=None,
        disjoint_classifier=None,
        prev_disjoint_classifier=None,
        diffusion_pretrained_dir=None,
        norm_grads=False,
        n_classes=10,
        random_generator=None,
        classifier_first_task_dir=None,
        train_noised_classifier=None,
    ):
        self.params = params
        self.task_id = task_id
        self.model = model
        self.prev_model = prev_model
        self.diffusion = diffusion
        self.data = data
        self.data_yielder = data_yielder
        self.data_loader = data_loader
        self.image_size = image_size
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.class_cond = class_cond
        self.use_task_index = use_task_index
        self.max_class = max_class
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.skip_save = skip_save
        self.plot_interval = plot_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.num_steps = num_steps
        self.scale_classes_loss = scale_classes_loss
        self.classes_per_task = classes_per_task
        self.use_ddim = use_ddim
        self.classifier_scale_min_old = (
            float(classifier_scale_min_old)
            if classifier_scale_min_old is not None
            else None
        )
        self.classifier_scale_max_old = (
            float(classifier_scale_max_old)
            if classifier_scale_max_old is not None
            else None
        )
        self.classifier_scale_min_new = (
            float(classifier_scale_min_new)
            if classifier_scale_min_new is not None
            else None
        )
        self.classifier_scale_max_new = (
            float(classifier_scale_max_new)
            if classifier_scale_max_new is not None
            else None
        )
        self.guid_generation_interval = guid_generation_interval
        self.use_old_grad = use_old_grad
        self.use_new_grad = use_new_grad
        self.guid_to_new_classes = guid_to_new_classes
        self.trim_logits = trim_logits
        self.disjoint_classifier = disjoint_classifier
        self.train_noised_classifier = train_noised_classifier
        self.prev_disjoint_classifier = prev_disjoint_classifier
        self.diffusion_pretrained_dir = diffusion_pretrained_dir
        self.train_transform_classifier = train_transform_classifier
        self.train_transform_diffusion = train_transform_diffusion
        self.norm_grads = norm_grads
        self.n_classes = n_classes
        self.random_generator = (
            random_generator
            if random_generator is not None
            else th.Generator().manual_seed(42)
        )

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        self.scheduler = th.optim.lr_scheduler.ExponentialLR(
            self.opt, gamma=scheduler_rate
        )
        self.scheduler_step = scheduler_step
        self.classifier_first_task_dir = classifier_first_task_dir

        if self.disjoint_classifier is not None:
            self.mp_trainer_classifier = MixedPrecisionTrainer(
                model=self.disjoint_classifier,
                use_fp16=self.use_fp16,
                initial_lg_loss_scale=16.0,
            )

            self.disjoint_classifier_optimizer = th.optim.SGD(
                self.mp_trainer_classifier.master_params,
                lr=(
                    self.params.classifier_lr
                    if self.task_id != 0
                    else self.params.classifier_init_lr
                ),
                weight_decay=self.params.classifier_weight_decay,
                momentum=0.9,
            )
            self.num_batches_per_epoch = len(self.data) // self.batch_size

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.prev_ddp_model = DDP(
                self.prev_model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
            self.prev_ddp_model.eval()
            if self.disjoint_classifier is not None:
                self.disjoint_classifier = DDP(
                    self.disjoint_classifier,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
                self.prev_disjoint_classifier = DDP(
                    self.prev_disjoint_classifier,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )
                self.prev_disjoint_classifier.eval()
            else:
                # NOTE: We need current diffusion only if we are not training classifier
                self.ddp_model = DDP(
                    self.model,
                    device_ids=[dist_util.dev()],
                    output_device=dist_util.dev(),
                    broadcast_buffers=False,
                    bucket_cap_mb=128,
                    find_unused_parameters=False,
                )

        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
            self.prev_ddp_model = self.prev_model

        self.diffusion_for_validation = prepare_diffusion(
            params, timestep_respacing=str(params.diffusion_steps_validation)
        )

        self.global_steps_before = global_steps_before

        self.cl_method = cl_method

        self.validation_loaders = validation_loaders

    def _load_and_sync_parameters(self):
        prev_resume_checkpoint = None
        curr_resume_checkpoint = None
        if self.resume_checkpoint:
            # continue training of the model on current task, start from previous task
            prev_resume_checkpoint = self.resume_checkpoint
            curr_resume_checkpoint = self.resume_checkpoint
            self.resume_step = parse_resume_step_from_filename(prev_resume_checkpoint)

        elif self.diffusion_pretrained_dir:
            # model already trained on current task
            all_checkpoints = os.listdir(self.diffusion_pretrained_dir)
            all_prev_checkpoints = [
                item
                for item in all_checkpoints
                if item.split("_")[-1] == f"{self.task_id - 1}.pt"
            ]
            all_prev_checkpoints = sorted(all_prev_checkpoints, reverse=True)
            if len(all_prev_checkpoints) > 1:  # take ema
                for ckpt in all_prev_checkpoints:
                    if "ema" in ckpt:
                        prev_resume_checkpoint = os.path.join(
                            self.diffusion_pretrained_dir, ckpt
                        )
                        break
            elif len(all_prev_checkpoints) == 1:
                prev_resume_checkpoint = os.path.join(
                    self.diffusion_pretrained_dir, all_prev_checkpoints[0]
                )
            all_curr_checkpoints = [
                item
                for item in all_checkpoints
                if item.split("_")[-1] == f"{self.task_id}.pt"
            ]
            all_curr_checkpoints = sorted(all_curr_checkpoints, reverse=True)
            if len(all_curr_checkpoints) > 1:  # take ema
                for ckpt in all_curr_checkpoints:
                    if "ema" in ckpt:
                        curr_resume_checkpoint = os.path.join(
                            self.diffusion_pretrained_dir, ckpt
                        )
                        break
            elif len(all_curr_checkpoints) == 1:
                curr_resume_checkpoint = os.path.join(
                    self.diffusion_pretrained_dir, all_curr_checkpoints[0]
                )

        if prev_resume_checkpoint is not None:
            if dist.get_rank() == 0:
                logger.log(
                    f"loading previous model from checkpoint: {prev_resume_checkpoint}..."
                )
            # Fix the forever loading of the model when training on multiple GPUs:
            # https://github.com/openai/guided-diffusion/issues/23#issuecomment-1055499214
            self.prev_model.load_state_dict(
                dist_util.load_state_dict(
                    prev_resume_checkpoint, map_location=dist_util.dev()
                )
            )
            dist_util.sync_params(self.prev_model.parameters())
            self.prev_model.eval()
        # NOTE: Only load the current model if we are not training classifier
        if curr_resume_checkpoint and self.class_cond:
            logger.log(
                f"loading current model from checkpoint: {curr_resume_checkpoint}..."
            )
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    curr_resume_checkpoint, map_location=dist_util.dev()
                )
            )
            dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        if self.disjoint_classifier is not None:
            print("Training disjoint classifier...")
            if (self.task_id == 0) and (self.classifier_first_task_dir is not None):
                self.disjoint_classifier.module.load_state_dict(
                    dist_util.load_state_dict(
                        self.classifier_first_task_dir, map_location=dist_util.dev()
                    )
                )
                dist_util.sync_params(self.disjoint_classifier.parameters())
            else:
                with tqdm(total=self.num_steps) as pbar:
                    epoch = 0
                    sampling_time = 0
                    prev_generations = None
                    prev_generations_labels = None
                    while self.step < self.num_steps:
                        self.step += 1
                        curr_epoch = self.step // self.num_batches_per_epoch
                        # Handle lr scheduling
                        if curr_epoch != epoch:
                            set_annealed_lr(
                                self.disjoint_classifier_optimizer,
                                (
                                    self.params.classifier_lr
                                    if self.task_id != 0
                                    else self.params.classifier_init_lr
                                ),
                                self.step / self.num_steps,
                            )
                            epoch = curr_epoch

                        if self.task_id > 0:
                            real_examples, real_cond = next(
                                self.data_yielder
                            )  # self.batch_size//(self.task_id+1)
                            if (
                                prev_generations is None
                                or (self.step - 1) % self.guid_generation_interval == 0
                            ):
                                # Generate replay examples and shuffle them with the real ones.
                                self.disjoint_classifier.eval()
                                sampling_start = time.time()
                                (
                                    generated_previous_examples,
                                    generated_previous_labels,
                                    generated_previous_examples_confidences,
                                ) = self.generate_examples(
                                    self.task_id - 1,
                                    (self.batch_size // (self.task_id + 1))
                                    * self.task_id,
                                    batch_size=-1,
                                    equal_n_examples_per_class=True,
                                    use_old_grad=self.use_old_grad,
                                    use_new_grad=self.use_new_grad,
                                    only_one_task=True,
                                    real_examples=real_examples,  # needed for speedup generation
                                    norm_grads=self.norm_grads,
                                )
                                sampling_time += time.time() - sampling_start
                                prev_generations = generated_previous_examples.cpu()
                                prev_generations_labels = (
                                    generated_previous_labels.cpu()
                                )
                            else:
                                generated_previous_examples = prev_generations
                                generated_previous_labels = prev_generations_labels
                            generated_previous_examples = (
                                generated_previous_examples.to(dist_util.dev())
                            )
                            generated_previous_labels = generated_previous_labels.to(
                                dist_util.dev()
                            )
                            cond = {"y": generated_previous_labels}
                            batch = th.cat([generated_previous_examples, real_examples])
                            cond = {
                                "y": th.cat(
                                    [
                                        generated_previous_labels,
                                        real_cond["y"].to(dist_util.dev()),
                                    ]
                                )
                            }
                            shuffle = th.randperm(batch.shape[0])
                            batch = batch[shuffle]
                            cond["y"] = cond["y"][shuffle]
                            if (
                                logger.get_rank_without_mpi_import() == 0
                                and (self.step - 1) % self.plot_interval == 0
                            ):
                                th.save(
                                    generated_previous_examples,
                                    os.path.join(
                                        get_blob_logdir(),
                                        f"generated_examples/generated_previous_examples_{self.task_id}_{self.step:06}.pt",
                                    ),
                                )
                                th.save(
                                    generated_previous_labels,
                                    os.path.join(
                                        get_blob_logdir(),
                                        f"generated_examples/generated_previous_labels_{self.task_id}_{self.step:06}.pt",
                                    ),
                                )
                            if (
                                logger.get_rank_without_mpi_import() == 0
                                and (self.step - 1) % self.log_interval == 0
                            ):
                                logger.log_generated_examples(
                                    generated_previous_examples,
                                    th.argmax(generated_previous_labels, 1),
                                    generated_previous_examples_confidences,
                                    self.task_id,
                                    n_examples_to_log=(
                                        (self.batch_size // (self.task_id + 1))
                                        * self.task_id
                                    )
                                    // (self.classes_per_task * self.task_id),
                                    step=self.get_global_step(),
                                )
                        else:
                            batch, cond = next(self.data_yielder)

                        self.disjoint_classifier.train()

                        # apply transforms here so that they are applied both to real images and generations
                        if self.train_transform_classifier is not None:
                            batch = self.train_transform_classifier(batch)

                        batch = batch.to(dist_util.dev())
                        y = cond["y"].to(dist_util.dev())
                        t = None

                        if self.train_noised_classifier:
                            t, _ = self.schedule_sampler.sample(
                                batch.shape[0], dist_util.dev()
                            )
                            batch = self.diffusion.q_sample(batch, t)

                        for i in range(0, batch.shape[0], self.microbatch):
                            micro = batch[i : i + self.microbatch].to(dist_util.dev())
                            micro_cond = y[i : i + self.microbatch].to(dist_util.dev())

                            replays_indices = th.where(
                                th.argmax(micro_cond, 1)
                                < (self.task_id) * self.classes_per_task
                            )[0]

                            out_classifier = self.disjoint_classifier(micro, t)
                            loss = F.cross_entropy(
                                out_classifier[:, : self.max_class + 1],
                                micro_cond[:, : self.max_class + 1],
                                reduction="none",
                            )

                        if (
                            logger.get_rank_without_mpi_import() == 0
                            and (self.step - 1) % self.log_interval == 0
                        ):
                            losses = {}
                            losses[f"disjoint_classifier_loss/{self.task_id}"] = (
                                loss.detach()
                            )
                            losses[
                                f"disjoint_classifier_loss_replay/{self.task_id}"
                            ] = loss.detach()[replays_indices]
                            losses[f"disjoint_classifier_loss_acc@1/{self.task_id}"] = (
                                compute_top_k(
                                    out_classifier,
                                    th.argmax(micro_cond, 1),
                                    k=1,
                                    reduction="none",
                                )
                            )
                            losses[f"disjoint_classifier_loss_acc@5/{self.task_id}"] = (
                                compute_top_k(
                                    out_classifier,
                                    th.argmax(micro_cond, 1),
                                    k=5,
                                    reduction="none",
                                )
                            )
                            wandb.log(
                                {k: v.mean().item() for k, v in losses.items()},
                                step=self.get_global_step(),
                            )
                            del losses

                        if i == 0:
                            self.mp_trainer_classifier.zero_grad()
                        loss = loss.mean()
                        self.mp_trainer_classifier.backward(
                            loss * len(micro) / len(batch)
                        )

                        self.mp_trainer_classifier.optimize(
                            self.disjoint_classifier_optimizer
                        )
                        pbar.update(1)

            if logger.get_rank_without_mpi_import() == 0:
                wandb.log(
                    {f"sampling_time/{self.task_id}": sampling_time},
                    step=self.get_global_step(),
                )
                print(f"sampling time: {sampling_time}")
            self.disjoint_classifier.eval()
            save_model(
                self.mp_trainer_classifier,
                self.disjoint_classifier_optimizer,
                self.step,
                self.task_id,
            )

        if not self.diffusion_pretrained_dir:
            self.ddp_model.train()
            print("Training diffusion...")
            with tqdm(total=self.num_steps) as pbar:
                while (
                    not self.lr_anneal_steps
                    or self.step + self.resume_step < self.lr_anneal_steps
                ) and (self.step < self.num_steps):
                    self.step += 1
                    pbar.update(1)
                    if self.step > 100:
                        self.mp_trainer.skip_gradient_thr = (
                            self.params.skip_gradient_thr
                        )
                    # apply transforms here so that they are applied both to real images and generations
                    batch, cond = next(self.data_yielder)
                    cond["y"] = th.argmax(cond["y"], 1)  # NOTE: Map from one-hot to int
                    if self.train_transform_diffusion is not None:
                        batch = self.train_transform_diffusion(batch)
                    self.run_step(batch, cond, self.step)
                    if (
                        (self.step - 1) % self.log_interval == 0
                        and logger.get_rank_without_mpi_import() == 0
                    ):
                        wandb_safe_log(logger.getkvs(), step=self.get_global_step())
                        logger.dumpkvs()
                    if (not self.skip_save) & (self.step % self.save_interval == 0):
                        self.save(self.task_id)
                        # Run for a finite amount of time in integration tests.
                        if (
                            os.environ.get("DIFFUSION_TRAINING_TEST", "")
                            and self.step > 0
                        ):
                            return
                    if self.step % self.scheduler_step == 0:
                        self.scheduler.step()
                # Save the last checkpoint if it wasn't already saved.
                if not self.skip_save:
                    if self.step % self.save_interval != 0:
                        self.save(self.task_id)

    def run_step(self, batch, cond, step):
        self.forward_backward(batch, cond, step)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond, step):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            # micro_cond = cond[i: i + self.microbatch].to(dist_util.dev())  # {
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            if isinstance(self.schedule_sampler, TaskAwareSampler):
                t, weights = self.schedule_sampler.sample(
                    micro.shape[0], dist_util.dev(), micro_cond["y"], self.task_id
                )
            else:
                t, weights = self.schedule_sampler.sample(
                    micro.shape[0], dist_util.dev()
                )

            # Do not scale the loss in the first task as the classes should be balanced
            if self.scale_classes_loss and self.task_id > 0:
                class_counts = th.bincount(
                    micro_cond["y"].long(), minlength=self.max_class + 1
                )
                class_weights = 1.0 / class_counts
                class_weights[th.isinf(class_weights)] = 0
                class_weights /= th.sum(class_weights)
                scaled_weights = weights * class_weights[micro_cond["y"]]
                weights = scaled_weights * (
                    th.sum(weights) / th.sum(scaled_weights)
                )  # scale such that the sum of weights does not change

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                self.task_id,
                model_kwargs=micro_cond,
                step=step,
                max_class=self.max_class + 1,
            )

            if last_batch or not self.use_ddp:
                losses, aux_info = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses, aux_info = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss_to_add, aux_losses = self.cl_method.get_additional_losses(
                self,
                x=micro,
                cond=micro_cond,
                t=t,
                x_t=aux_info["x_t"],
                t_scaled=aux_info["t_scaled"],
            )
            loss = (losses["loss"] * weights).mean() + loss_to_add
            losses.update(aux_losses)

            # log loss of replay examples
            replays_indices = th.where(
                micro_cond["y"] < (self.task_id) * self.classes_per_task
            )[0]
            replay_loss = []
            for i in range(len(losses["loss"])):
                if i in replays_indices:
                    replay_loss.append(losses["loss"][i])
                else:
                    replay_loss.append(0)
            losses["loss_replay"] = th.tensor(replay_loss).to(dist_util.dev())

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step_cur_task", self.step + self.resume_step)
        logger.logkv(
            "samples_cur_task", (self.step + self.resume_step) * self.global_batch
        )

    def save(self, task_id):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}_{task_id}.pt"
                else:
                    filename = (
                        f"ema_{rate}_{(self.step + self.resume_step):06d}_{task_id}.pt"
                    )
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(
                    get_blob_logdir(), f"opt{(self.step + self.resume_step):06d}.pt"
                ),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _sample_examples(
        self,
        total_num_examples,
        batch_size,
        max_class_generate,
        diffusion,
        use_old_grad=True,
        use_new_grad=True,
        trim_logits=False,
        max_class=None,
        real_examples=None,
        norm=False,
    ):
        all_images = []
        all_labels = []
        if max_class is None:
            max_class = self.max_class

        # NOTE: When class_cond=True, classifier guidance will not be used
        if (not self.class_cond) and (use_old_grad or use_new_grad):
            # prepare models
            old_classifier_fn = self.prev_disjoint_classifier
            if use_new_grad:
                new_classifier_fn = self.disjoint_classifier

        # NOTE: Possible further improvements from http://arxiv.org/abs/2302.07121, but with them
        # sampling becomes very time-consuming.
        def cond_fn(x, t, pred_xstart, y=None):
            assert y is not None
            with th.enable_grad():
                if not use_old_grad and not use_new_grad:
                    return th.zeros_like(x)
                if use_old_grad and use_new_grad:
                    raise NotImplementedError()

                if self.train_noised_classifier:
                    x_in = x.detach().requires_grad_(True)
                    x_out = x_in
                else:
                    x_in = pred_xstart
                    x_out = x

                if use_old_grad:
                    logits_old = old_classifier_fn(x_in, t)
                    if self.trim_logits:
                        logits_old = logits_old[
                            :, : max_class - self.classes_per_task + 1
                        ]
                    loss_old = -F.cross_entropy(logits_old, y, reduction="none")
                    if self.params.negate_old_grad:
                        loss_old = -loss_old
                    grad_old = th.autograd.grad(loss_old.sum(), x_out)[0]
                    if norm:
                        norm_val = th.linalg.norm(grad_old.flatten(1), ord=th.inf)
                        grad_old = grad_old / norm_val
                    if not use_new_grad:
                        return grad_old * classfier_scale_vec_old.view(-1, 1, 1, 1)
                if use_new_grad:
                    logits_new = new_classifier_fn(x_in, t)
                    if self.trim_logits:
                        logits_new = logits_new[:, : max_class + 1]

                    if self.guid_to_new_classes:
                        probs = F.softmax(
                            logits_new[
                                :,
                                max_class + 1 - self.classes_per_task : max_class + 1,
                            ],
                            dim=-1,
                        )
                        random_new_task_classes = (
                            th.argmax(probs, dim=-1)
                            + max_class
                            + 1
                            - self.classes_per_task
                        )
                        loss_new = -F.cross_entropy(
                            logits_new, random_new_task_classes, reduction="none"
                        )
                    else:
                        loss_new = F.cross_entropy(logits_new, y, reduction="none")
                    grad_new = th.autograd.grad(loss_new.sum(), x_out)[0]
                    if norm:
                        norm_val = th.linalg.norm(grad_new.flatten(1), ord=th.inf)
                        grad_new = grad_new / norm_val
                    if use_old_grad:
                        return grad_old * classfier_scale_vec_old.view(
                            -1, 1, 1, 1
                        ) + grad_new * classfier_scale_vec_new.view(-1, 1, 1, 1)
                    else:
                        return grad_new * classfier_scale_vec_new.view(-1, 1, 1, 1)

        with tqdm(total=total_num_examples, leave=False) as progress_bar:
            while len(all_images) * batch_size < total_num_examples:
                model_kwargs = {}
                classes = (
                    th.randint(0, max_class_generate, size=(batch_size,))
                    .long()
                    .to(dist_util.dev())
                )

                model_kwargs["y"] = classes

                if not self.class_cond and use_old_grad:
                    classfier_scale_vec_old = (
                        th.from_numpy(
                            np.random.uniform(
                                low=self.classifier_scale_min_old,
                                high=self.classifier_scale_max_old,
                                size=(len(classes),),
                            )
                        )
                        .float()
                        .to(dist_util.dev())
                    )
                if not self.class_cond and use_new_grad:
                    classfier_scale_vec_new = (
                        th.from_numpy(
                            np.random.uniform(
                                low=self.classifier_scale_min_new,
                                high=self.classifier_scale_max_new,
                                size=(len(classes),),
                            )
                        )
                        .float()
                        .to(dist_util.dev())
                    )
                sample_fn = (
                    diffusion.ddim_sample_loop
                    if self.use_ddim
                    else diffusion.p_sample_loop
                )
                sample = sample_fn(
                    self.prev_ddp_model,
                    (
                        len(classes),
                        self.in_channels,
                        self.image_size,
                        self.image_size,
                    ),
                    clip_denoised=self.params.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=(None if self.class_cond else cond_fn),
                    device=dist_util.dev(),
                    compute_grads=(
                        not self.class_cond
                        and (
                            not self.train_noised_classifier
                            and (use_old_grad or use_new_grad)
                        )
                    ),
                )

                sample = sample.detach()
                if self.class_cond:
                    sample = sample.contiguous()
                    gathered_samples = [
                        th.zeros_like(sample) for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(
                        gathered_samples, sample
                    )  # gather not supported with NCCL
                    all_images.extend([sample.cpu() for sample in gathered_samples])
                    gathered_labels = [
                        th.zeros_like(classes) for _ in range(dist.get_world_size())
                    ]
                    dist.all_gather(gathered_labels, classes)
                    all_labels.extend(
                        [
                            th.eye(self.n_classes)[labels.cpu()]
                            for labels in gathered_labels
                        ]
                    )

                    progress_bar.update(len(sample) * dist.get_world_size())
                else:
                    all_images.extend([sample.cpu()])
                    all_labels.extend([th.eye(self.n_classes)[classes.cpu()]])
                    progress_bar.update(len(sample))

                th.cuda.empty_cache()

        all_images = th.cat(all_images)
        all_labels = th.cat(all_labels)
        if self.class_cond:
            dist.barrier()

        return all_images, all_labels, None

    # @th.no_grad()
    def generate_examples(
        self,
        task_id,
        n_examples_per_task,
        batch_size=-1,
        only_one_task=False,
        use_diffusion_for_validation=False,
        equal_n_examples_per_class=False,
        use_old_grad=True,
        use_new_grad=True,
        trim_logits=False,
        max_class=None,
        real_examples=None,
        norm_grads=False,
    ):
        if not only_one_task:
            total_num_examples = n_examples_per_task * (task_id + 1)
        else:
            total_num_examples = n_examples_per_task

        if batch_size == -1:
            batch_size = total_num_examples

        if use_diffusion_for_validation:
            diffusion = self.diffusion_for_validation
        else:
            diffusion = self.diffusion

        (
            all_images,
            all_labels,
            all_confidences,
        ) = self._sample_examples(
            total_num_examples,
            batch_size,
            (task_id + 1) * self.classes_per_task,
            diffusion,
            use_old_grad=use_old_grad,
            use_new_grad=use_new_grad,
            trim_logits=trim_logits,
            max_class=max_class,
            real_examples=real_examples,
            norm=norm_grads,
        )

        return all_images, all_labels, None

    def get_global_step(self):
        return self.global_steps_before + self.step

    def append_generated_data(self, new_examples, new_labels):
        generated_dataset = AppendName(
            TensorDataset(new_examples, new_labels),
            new_labels.cpu().numpy(),
            True,
            False,
        )
        joined_dataset = ConcatDataset([self.data, generated_dataset])
        train_dataset_loader = DataLoader(
            dataset=joined_dataset,
            batch_size=(self.params.batch_size // int(os.environ["WORLD_SIZE"])),
            shuffle=True,
            drop_last=True,
        )
        self.data_yielder = yielder(train_dataset_loader)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        if key != "prev_kl" and key != "replay_loss":
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step, task_id):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"disjoint_clf_{task_id}.pt"),
        )
        th.save(
            opt.state_dict(),
            os.path.join(logger.get_dir(), f"disjoint_clf_opt_{task_id}.pt"),
        )


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def prepare_diffusion(args, timestep_respacing):
    return create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        sigma_small=args.sigma_small,
        noise_schedule=args.noise_schedule,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
