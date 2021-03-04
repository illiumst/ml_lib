from pytorch_lightning import LightningDataModule


# Dataset Options
from ml_lib.utils.tools import add_argparse_args

DATA_OPTION_test = 'test'
DATA_OPTION_devel = 'devel'
DATA_OPTION_train = 'train'
DATA_OPTIONS = [DATA_OPTION_train, DATA_OPTION_devel, DATA_OPTION_test]


class _BaseDataModule(LightningDataModule):

    @property
    def shape(self):
        return self.datasets[DATA_OPTION_train].sample_shape

    @classmethod
    def add_argparse_args(cls, parent_parser):
        return add_argparse_args(cls, parent_parser)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasets = dict()

    def transfer_batch_to_device(self, batch, device):
        if isinstance(batch, list):
            for idx, item in enumerate(batch):
                try:
                    batch[idx] = item.to(device)
                except (AttributeError, RuntimeError):
                    continue
            return batch
        else:
            return batch.to(device)
