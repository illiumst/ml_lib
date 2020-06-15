from typing import Union

import torch
from torch_geometric.data import Data


class BatchToData(object):
    def __init__(self, transforms=None):
        super(BatchToData, self).__init__()
        self.transforms = transforms if transforms else lambda x: x

    def __call__(self, batch_norm: torch.Tensor, batch_pos: torch.Tensor,
                 batch_y_l: Union[torch.Tensor, None] = None, batch_y_c: Union[torch.Tensor, None] = None):
        # Convert to torch_geometric.data.Data type
        # data = data.transpose(1, 2).contiguous()
        batch_size, num_points, _ = batch_norm.shape  # (batch_size, num_points, 3)

        norm = batch_norm.reshape(batch_size * num_points, -1)
        pos = batch_pos.reshape(batch_size * num_points, -1)
        batch_y_l = batch_y_l.reshape(batch_size * num_points) if batch_y_l is not None else batch_y_l
        batch_y_c = batch_y_c.reshape(batch_size * num_points) if batch_y_c is not None else batch_y_c
        batch = torch.zeros((batch_size, num_points), device=pos.device, dtype=torch.long)
        for i in range(batch_size):
            batch[i] = i
        batch = batch.view(-1)

        data = Data()
        data.norm, data.pos, data.batch, data.yl, data.yc = norm, pos, batch, batch_y_l, batch_y_c

        data = self.transforms(data)

        return data
