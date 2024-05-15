import torch as th
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from cl_methods.base import CLMethod
from dataloaders.utils import yielder
from dataloaders.wrapper import AppendName
from guide import dist_util
from guide.logger import get_rank_without_mpi_import, log_generated_examples


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
