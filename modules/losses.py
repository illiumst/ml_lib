from typing import List

import torch
from torch import nn

from ml_lib.modules.utils import FlipTensor
from ml_lib.objects.map import MapStorage, Map
from ml_lib.objects.trajectory import Trajectory


class BinaryHomotopicLoss(nn.Module):
    def __init__(self, map_storage: MapStorage):
        super(BinaryHomotopicLoss, self).__init__()
        self.map_storage = map_storage
        self.flipper = FlipTensor()

    def forward(self, x: torch.Tensor, y: torch.Tensor, mapnames: str):
        maps: List[Map] = [self.map_storage[mapname] for mapname in mapnames]
        for basemap in maps:
            basemap = basemap.as_2d_array



