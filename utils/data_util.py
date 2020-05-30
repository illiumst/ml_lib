import torch
from torch.utils.data import Dataset


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


class ReMapDataset(Dataset):
    @property
    def sample_shape(self):
        return list(self[0][0].shape)

    def __init__(self, ds, mapping):
        super(ReMapDataset, self).__init__()
        # here is a mapping from this index to the mother ds index
        self.mapping = mapping
        self.ds = ds

    def __getitem__(self, index):
        return self.ds[self.mapping[index]]

    def __len__(self):
        return self.mapping.shape[0]

    @classmethod
    def do_train_vali_split(cls, ds, split_fold=0.1):

        indices = torch.randperm(len(ds))

        valid_size = int(len(ds) * split_fold)

        train_mapping = indices[valid_size:]
        valid_mapping = indices[:valid_size]

        train = cls(ds, train_mapping)
        valid = cls(ds, valid_mapping)

        return train, valid