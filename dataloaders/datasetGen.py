import torch
from torch.utils.data import Subset

from .utils import get_stratified_subset
from .wrapper import AppendName
import numpy as np
from collections import defaultdict


def data_split(
    dataset,
    return_classes=False,
    return_task_as_class=False,
    num_tasks=5,
    num_classes=10,
    limit_classes=-1,
    validation_frac=0.3,
    data_seed=0,
    shared_classes=False,
    first_task_num_classes=0,
):
    train_dataset_splits = {}
    val_dataset_splits = {}

    if not shared_classes:
        if limit_classes > 0:
            assert limit_classes <= num_classes
            num_classes = limit_classes

        # assert num_classes % num_tasks == 0
        if first_task_num_classes > 0:
            classes_per_task = (num_classes - first_task_num_classes) // (num_tasks - 1)
            class_split = {
                i: list(
                    range(
                        ((i - 1) * classes_per_task) + first_task_num_classes,
                        (i * classes_per_task) + first_task_num_classes,
                    )
                )
                for i in range(1, num_tasks)
            }
            class_split[0] = list(range(0, first_task_num_classes))
            class_split = dict(sorted(class_split.items()))
        else:
            classes_per_task = num_classes // num_tasks
            class_split = {
                i: list(range(i * classes_per_task, (i + 1) * classes_per_task))
                for i in range(num_tasks)
            }
        labels = (
            dataset.labels
            if dataset.labels.shape[1] == 1
            else torch.argmax(dataset.labels, 1)
        )
        class_indices = torch.LongTensor(labels)
        task_indices_1hot = torch.zeros(
            len(dataset), num_tasks
        )  # 1hot array describing to which tasks datapoints belong.
        for task, classes in class_split.items():
            task_indices_1hot[
                (class_indices[..., None] == torch.tensor(classes)).any(-1), task
            ] = 1

        train_set_indices_bitmask = torch.ones(len(dataset))
        validation_indices = get_stratified_subset(
            validation_frac, labels, seed=data_seed
        )
        train_set_indices_bitmask[validation_indices] = 0

        for task, classes in class_split.items():
            cur_task_indices_bitmask = task_indices_1hot[:, task] == 1
            cur_train_indices_bitmask = (
                train_set_indices_bitmask * cur_task_indices_bitmask
            )
            cur_val_indices_bitmask = (
                1 - train_set_indices_bitmask
            ) * cur_task_indices_bitmask

            train_subset = Subset(
                dataset, torch.where(cur_train_indices_bitmask == 1)[0]
            )
            train_subset.class_list = classes

            val_subset = Subset(dataset, torch.where(cur_val_indices_bitmask == 1)[0])
            val_subset.class_list = classes

            train_dataset_splits[task] = AppendName(
                train_subset,
                [task] * len(train_subset),
                return_classes=return_classes,
                return_task_as_class=return_task_as_class,
            )
            val_dataset_splits[task] = AppendName(
                val_subset,
                [task] * len(train_subset),
                return_classes=return_classes,
                return_task_as_class=return_task_as_class,
            )

    else:
        # Each class in every task
        class_examples = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            class_examples[label].append(idx)

        classes_per_task = num_classes

        # Calculate the number of examples per class for each part and train/validation sets
        examples_per_class_per_part = len(class_examples[0]) // num_tasks
        train_examples = int(examples_per_class_per_part * (1 - validation_frac))
        val_examples = examples_per_class_per_part - train_examples
        for task in range(num_tasks):
            train_indices = []
            val_indices = []

            for class_idx in class_examples.keys():
                start_idx = task * examples_per_class_per_part
                train_end_idx = start_idx + train_examples
                val_end_idx = train_end_idx + val_examples

                train_indices.extend(class_examples[class_idx][start_idx:train_end_idx])
                val_indices.extend(class_examples[class_idx][train_end_idx:val_end_idx])

            train_subset = Subset(dataset, train_indices)
            train_subset.class_list = list(range(num_classes))
            val_subset = Subset(dataset, val_indices)
            val_subset.class_list = list(range(num_classes))

            train_dataset_splits[task] = AppendName(
                train_subset,
                [task] * len(train_subset),
                return_classes=return_classes,
                return_task_as_class=return_task_as_class,
            )
            val_dataset_splits[task] = AppendName(
                val_subset,
                [task] * len(train_subset),
                return_classes=return_classes,
                return_task_as_class=return_task_as_class,
            )

    print(
        f"Prepared dataset with splits: {[(idx, len(data)) for idx, data in enumerate(train_dataset_splits.values())]}"
    )
    print(
        f"Validation dataset with splits: {[(idx, len(data)) for idx, data in enumerate(val_dataset_splits.values())]}"
    )
    if hasattr(dataset.dataset, "classes"):
        print(
            f"Prepared class order: {[(idx, [np.array(dataset.dataset.classes)[data.dataset.class_list]]) for idx, data in enumerate(train_dataset_splits.values())]}"
        )

    return train_dataset_splits, val_dataset_splits, classes_per_task
