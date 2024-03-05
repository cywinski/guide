from torch.utils.data import DataLoader

from cl_methods.base import CLMethod
from dataloaders.utils import yielder


class GenerativeReplayDisjointClassifierGuidance(CLMethod):
    def get_data_for_task(
        self,
        dataset,
        task_id,
        train_loop,
        generator=None,
        step=None,
    ):
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size // (task_id + 1),
            shuffle=True,
            drop_last=True,
            generator=generator,
        )
        return yielder(loader), loader, None
