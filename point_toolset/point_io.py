from typing import Union

import torch
from torch_geometric.data import Data


class BatchToData(object):
    def __init__(self, transforms=None):
        super(BatchToData, self).__init__()
        self.transforms = transforms if transforms else lambda x: x

    def __call__(self, batch_dict):
        # Convert to torch_geometric.data.Data type

        batch_pos = batch_dict['pos']
        batch_norm = batch_dict['norm']
        batch_y = batch_dict['y']
        batch_y_c = batch_dict['y_c']

        batch_size, num_points, _ = batch_pos.shape  # (batch_size, num_points, 3)

        batch_size, N, _ = batch_pos.shape  # (batch_size, num_points, 3)
        pos = batch_pos.view(batch_size * N, -1)
        norm = batch_norm.view(batch_size * N, -1) if batch_norm is not None else batch_norm

        batch_y_l = batch_y.view(batch_size * N, -1) if batch_y is not None else batch_y
        batch_y_c = batch_y_c.view(batch_size * N, -1) if batch_y_c is not None else batch_y_c

        batch = torch.zeros((batch_size, num_points), device=pos.device, dtype=torch.long)
        for i in range(batch_size):
            batch[i] = i
        batch = batch.view(-1)

        data = Data()
        data.norm, data.pos, data.batch, data.yl, data.yc = norm, pos, batch, batch_y_l, batch_y_c

        data = self.transforms(data)

        return data
