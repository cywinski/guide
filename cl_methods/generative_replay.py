import torch as th
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from cl_methods.base import CLMethod
from dataloaders.utils import yielder
from dataloaders.wrapper import AppendName
from guided_diffusion import dist_util
from guided_diffusion.logger import get_rank_without_mpi_import, log_generated_examples


class GenerativeReplay(CLMethod):
    def get_data_for_task(
        self,
        dataset,
        task_id,
        train_loop,
        generator=None,
        step=None,
    ):
        if task_id == 0:
            train_dataset_loader = DataLoader(
                dataset=dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                drop_last=True,
                generator=generator,
            )
            dataset_yielder = yielder(train_dataset_loader)
        else:
            print("Preparing dataset for rehearsal...")
            if self.args.gr_n_generated_examples_per_task <= self.args.batch_size:
                batch_size = self.args.gr_n_generated_examples_per_task
            else:
                batch_size = self.args.batch_size
            (
                generated_previous_examples,
                generated_previous_examples_labels,
                generated_previous_examples_confidences,
            ) = train_loop.generate_examples(
                task_id - 1,
                self.args.gr_n_generated_examples_per_task,
                batch_size=batch_size,
                equal_n_examples_per_class=True,
                use_old_grad=False,
                use_new_grad=False,
            )
            print("moving generated dataset to gpu...")
            generated_previous_examples = generated_previous_examples.to(
                dist_util.dev()
            )
            generated_previous_examples_labels = generated_previous_examples_labels.to(
                dist_util.dev()
            )
            generated_dataset = AppendName(
                TensorDataset(
                    generated_previous_examples, generated_previous_examples_labels
                ),
                generated_previous_examples_labels.cpu().numpy(),
                True,
                False,
            )
            joined_dataset = ConcatDataset([dataset, generated_dataset])
            train_dataset_loader = DataLoader(
                dataset=joined_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                drop_last=True,
                generator=generator,
            )
            dataset_yielder = yielder(train_dataset_loader)
            if get_rank_without_mpi_import() == 0:
                log_generated_examples(
                    generated_previous_examples,
                    th.argmax(generated_previous_examples_labels, 1),
                    generated_previous_examples_confidences,
                    task_id,
                    step=step,
                )

        return (
            dataset_yielder,
            train_dataset_loader,
            generated_previous_examples if task_id != 0 else None,
        )

    def get_additional_losses(self, train_loop, x, cond, t, x_t, t_scaled):
        if self.args.gr_generate_previous_samples_continuously and (
            train_loop.task_id > 0
        ):
            shape = [
                train_loop.batch_size,
                train_loop.in_channels,
                train_loop.image_size,
                train_loop.image_size,
            ]
            prev_loss = train_loop.diffusion.calculate_loss_previous_task(
                current_model=train_loop.ddp_model,
                prev_model=train_loop.prev_ddp_model,  # Frozen copy of the model
                schedule_sampler=train_loop.schedule_sampler,
                task_id=train_loop.task_id,
                n_examples_per_task=train_loop.batch_size,
                shape=shape,
                batch_size=train_loop.microbatch,
                clip_denoised=train_loop.params.clip_denoised,
            )

            return prev_loss, {"prev_loss": prev_loss}
        else:
            return 0, {}
