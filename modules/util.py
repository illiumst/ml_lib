from abc import ABC
from pathlib import Path

import torch
from torch import nn
from torch import functional as F

import pytorch_lightning as pl


# Utility - Modules
###################
from ..utils.model_io import ModelParameters


class LightningBaseModule(pl.LightningModule, ABC):

    @classmethod
    def name(cls):
        return cls.__name__

    @property
    def shape(self):
        try:
            x = torch.randn(self.in_shape).unsqueeze(0)
            output = self(x)
            return output.shape[1:]
        except Exception as e:
            print(e)
            return -1

    def __init__(self, hparams):
        super(LightningBaseModule, self).__init__()

        # Set Parameters
        ################################
        self.hparams = hparams
        self.params = ModelParameters(hparams)

        # Dataset Loading
        ################################
        # TODO: Find a way to push Class Name, library path and parameters (sometimes thiose are objects) in here

    def size(self):
        return self.shape

    def save_to_disk(self, model_path):
        Path(model_path, exist_ok=True).mkdir(parents=True, exist_ok=True)
        if not (model_path / 'model_class.obj').exists():
            with (model_path / 'model_class.obj').open('wb') as f:
                torch.save(self.__class__, f)
        return True

    @property
    def data_len(self):
        return len(self.dataset.train_dataset)

    @property
    def n_train_batches(self):
        return len(self.train_dataloader())

    def configure_optimizers(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, batch_xy, batch_nb, *args, **kwargs):
        raise NotImplementedError

    def test_step(self, *args, **kwargs):
        raise NotImplementedError

    def test_epoch_end(self, outputs):
        raise NotImplementedError

    def init_weights(self, in_place_init_func_=nn.init.xavier_uniform_):
        weight_initializer = WeightInit(in_place_init_function=in_place_init_func_)
        self.apply(weight_initializer)


class ShapeMixin:

    @property
    def shape(self):
        assert isinstance(self, (LightningBaseModule, nn.Module))
        if self.in_shape is not None:
            x = torch.randn(self.in_shape)
            # This is needed for BatchNorm shape checking
            x = torch.stack((x, x))
            output = self(x)
            return output.shape[1:] if len(output.shape[1:]) > 1 else output.shape[-1]
        else:
            return -1


class F_x(ShapeMixin, nn.Module):
    def __init__(self, in_shape):
        super(F_x, self).__init__()
        self.in_shape = in_shape

    def forward(self, x):
        return x


# Utility - Modules
###################
class Flatten(ShapeMixin, nn.Module):

    def __init__(self, in_shape, to=-1):
        assert isinstance(to, int) or isinstance(to, tuple)
        super(Flatten, self).__init__()
        self.in_shape = in_shape
        self.to = (to,) if isinstance(to, int) else to

    def forward(self, x):
        return x.view(x.size(0), *self.to)


class Interpolate(nn.Module):
    def __init__(self,  size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x


class AutoPad(nn.Module):

    def __init__(self, interpolations=3, base=2):
        super(AutoPad, self).__init__()
        self.fct = base ** interpolations

    def forward(self, x):
        # noinspection PyUnresolvedReferences
        x = F.pad(x,
                  [0,
                   (x.shape[-1] // self.fct + 1) * self.fct - x.shape[-1] if x.shape[-1] % self.fct != 0 else 0,
                   (x.shape[-2] // self.fct + 1) * self.fct - x.shape[-2] if x.shape[-2] % self.fct != 0 else 0,
                   0])
        return x


class WeightInit:

    def __init__(self, in_place_init_function):
        self.in_place_init_function = in_place_init_function

    def __call__(self, m):
        if hasattr(m, 'weight'):
            if isinstance(m.weight, torch.Tensor):
                if m.weight.ndim < 2:
                    m.weight.data.fill_(0.01)
                else:
                    self.in_place_init_function(m.weight)
        if hasattr(m, 'bias'):
            if isinstance(m.bias, torch.Tensor):
                m.bias.data.fill_(0.01)


class FilterLayer(nn.Module):

    def __init__(self):
        super(FilterLayer, self).__init__()

    def forward(self, x):
        tensor = x[:, -1]
        return tensor


class MergingLayer(nn.Module):

    def __init__(self):
        super(MergingLayer, self).__init__()

    def forward(self, x):
        # ToDo: Which ones to combine?
        return


class FlipTensor(nn.Module):
    def __init__(self, dim=-2):
        super(FlipTensor, self).__init__()
        self.dim = dim

    def forward(self, x):
        idx = [i for i in range(x.size(self.dim) - 1, -1, -1)]
        idx = torch.as_tensor(idx).long()
        inverted_tensor = x.index_select(self.dim, idx)
        return inverted_tensor


class AutoPadToShape(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.shape[1:] == self.shape:
            return x
        embedding = torch.zeros((x.shape[0], *self.shape))
        embedding[:, :x.shape[1], :x.shape[2], :x.shape[3]] = x
        return embedding

    def __repr__(self):
        return f'AutoPadTransform({self.shape})'


class HorizontalSplitter(nn.Module):

    def __init__(self, in_shape, n):
        super(HorizontalSplitter, self).__init__()
        assert len(in_shape) == 3
        self.n = n
        self.in_shape = in_shape

        self.channel, self.height, self.width = self.in_shape
        self.new_height = (self.height // self.n) + (1 if self.height % self.n != 0 else 0)

        self.shape = (self.channel, self.new_height, self.width)
        self.autopad = AutoPadToShape(self.shape)

    def forward(self, x):
        n_blocks = list()
        for block_idx in range(self.n):
            start = block_idx * self.new_height
            end = (block_idx + 1) * self.new_height
            block = self.autopad(x[:, :, start:end, :])
            n_blocks.append(block)

        return n_blocks


class HorizontalMerger(nn.Module):

    @property
    def shape(self):
        merged_shape = self.in_shape[0], self.in_shape[1] * self.n, self.in_shape[2]
        return merged_shape

    def __init__(self, in_shape, n):
        super(HorizontalMerger, self).__init__()
        assert len(in_shape) == 3
        self.n = n
        self.in_shape = in_shape

    def forward(self, x):
        return torch.cat(x, dim=-2)
