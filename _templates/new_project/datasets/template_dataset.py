from torch.utils.data import Dataset


class TemplateDataset(Dataset):

    @property
    def sample_shape(self):
        return self[0][0].shape

    def __init__(self, *args, **kwargs):
        super(TemplateDataset, self).__init__()

    def __len__(self):
        pass

    def __getitem__(self, item):
        return item
