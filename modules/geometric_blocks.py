import torch
from torch import nn
from torch.nn import ReLU

from torch_geometric.nn import PointConv, fps, radius, global_max_pool, knn_interpolate


class SAModule(torch.nn.Module):

    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(nn.Module):
    def __init__(self, nn, channels=3):
        super(GlobalSAModule, self).__init__()
        self.nn = nn
        self.channels = channels

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), self.channels))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class MLP(nn.Module):
    def __init__(self, channels, norm=True):
        super(MLP, self).__init__()
        self.net = nn.Sequential(*[
            nn.Sequential(nn.Linear(channels[i - 1], channels[i]), ReLU(), nn.BatchNorm1d(channels[i]))
            for i in range(1, len(channels))
        ]).double()

    def forward(self, x, *args, **kwargs):
        return self.net(x)


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip