import torch
from torch_geometric.data import Data


class BatchToData(object):
    def __init__(self):
        super(BatchToData, self).__init__()

    def __call__(self, batch_x: torch.Tensor, batch_pos: torch.Tensor, batch_y: torch.Tensor):
        # Convert to torch_geometric.data.Data type
        # data = data.transpose(1, 2).contiguous()
        batch_size, num_points, _ = batch_x.shape  # (batch_size, num_points, 3)

        x = batch_x.reshape(batch_size * num_points, -1)
        pos = batch_pos.reshape(batch_size * num_points, -1)
        batch_y = batch_y.reshape(batch_size * num_points)
        batch = torch.zeros((batch_size, num_points), device=pos.device, dtype=torch.long)
        for i in range(batch_size):
            batch[i] = i
        batch = batch.view(-1)

        data = Data()
        data.x, data.pos, data.batch, data.y = x, pos, batch, batch_y
        return data
