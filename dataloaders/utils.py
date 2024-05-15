import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader


def yielder(loader):
    while True:
        yield from loader


def concat_yielder(loaders):
    iterators = [iter(loader) for loader in loaders]
    while True:
        result = []
        for i in range(len(iterators)):
            try:
                x = next(iterators[i])
            except StopIteration:
                iterators[i] = iter(loaders[i])
                x = next(iterators[i])
            result.append(x)

        yield recursive_concat(result)


def recursive_concat(l):
    """Concat elements in list l. Each element can be a tensor, dicts of tensors, tuple, etc."""
    if isinstance(l[0], torch.Tensor):
        return torch.cat(l)
    if isinstance(l[0], dict):
        keys = set(l[0].keys())
        for x in l[1:]:
            assert set(x.keys()) == keys
        return {k: recursive_concat([x[k] for x in l]) for k in keys}
    if isinstance(l[0], tuple) or isinstance(l[0], list):
        length = len(l[0])
        for x in l[1:]:
            assert len(x) == length
        return tuple([recursive_concat([x[i] for x in l]) for i in range(length)])


def get_stratified_subset(frac_selected, labels, seed=0):
    """Returns indices of a subset with a given percentage of elements for each class."""
    labels = np.array(labels)
    rng = np.random.default_rng(seed=seed)
    res = []
    for l in np.unique(labels):
        all_indices = np.nonzero(labels == l)[0]
        num_selected = int(frac_selected * len(all_indices))
        res.append(rng.choice(all_indices, num_selected, replace=False))
    res = np.concatenate(res)
    return res


def prepare_eval_loaders(
    train_dataset_splits, val_dataset_splits, args, include_train, generator=False
):
    eval_loaders = []
    for task_id in range(args.num_tasks):
        if include_train:
            eval_data = ConcatDataset(
                [train_dataset_splits[task_id], val_dataset_splits[task_id]]
            )
        else:
            eval_data = val_dataset_splits[task_id]
        eval_loader = DataLoader(
            dataset=eval_data,
            batch_size=args.batch_size,
            shuffle=False,
            generator=generator,
        )
        eval_loaders.append(eval_loader)

    return eval_loaders
