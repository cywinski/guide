"""CL-method-specific data preparation."""

from torch.utils.data import DataLoader

from dataloaders.utils import yielder


class CLMethod:
    def __init__(self, args):
        self.args = args

    def get_data_for_task(self, dataset, task_id, train_loop):
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        return yielder(loader), loader, None

    def get_additional_losses(self, train_loop, x, cond, t, x_t, t_scaled):
        """Returns: (scalar, dict {name: scalar}), meaning (loss to add to the objective, additional losses)."""
        return 0, {}
