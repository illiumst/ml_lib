import torch
from torch_geometric.transforms import NormalizeScale


class NormalizePositions(NormalizeScale):

    def __init__(self):
        super(NormalizePositions, self).__init__()

    def __call__(self, data):
        if torch.isnan(data.pos).any():
            print('debug')

        data = self.center(data)
        if torch.isnan(data.pos).any():
            print('debug')

        scale = (1 / data.pos.abs().max()) * 0.999999
        if torch.isnan(scale).any() or torch.isinf(scale).any():
            print('debug')

        data.pos = data.pos * scale
        if torch.isnan(data.pos).any():
            print('debug')

        return data
