from functools import reduce

from abc import ABC
from pathlib import Path

import torch
from operator import mul
from torch import nn
from torch.nn import functional as F, Unfold

# Utility - Modules
###################
from ..utils.model_io import ModelParameters
from ..utils.tools import locate_and_import_class

try:
    import pytorch_lightning as pl

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
            self.lr = self.params.lr or 1e-4

        def size(self):
            return self.shape

        def additional_scores(self, outputs):
            raise NotImplementedError

        @property
        def dataset_class(self):
            try:
                return locate_and_import_class(self.params.class_name, folder_path='datasets')
            except AttributeError as e:
                raise AttributeError(f'The dataset alias you provided ("{self.params.class_name}") ' +
                                     f'was not found!\n' +
                                     f'{e}')

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

    module_types = (LightningBaseModule, nn.Module,)

except ImportError:
    module_types = (nn.Module,)
    pl = None
    pass  # Maybe post a hint to install pytorch-lightning.


class ShapeMixin:

    @property
    def shape(self):

        assert isinstance(self, module_types)

        def get_out_shape(output):
            return output.shape[1:] if len(output.shape[1:]) > 1 else output.shape[-1]

        in_shape = self.in_shape if hasattr(self, 'in_shape') else None
        if in_shape is not None:
            try:
                device = self.device
            except AttributeError:
                try:
                    device = next(self.parameters()).device
                except StopIteration:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
            x = torch.randn(in_shape, device=device)
            # This is needed for BatchNorm shape checking
            x = torch.stack((x, x))

            # noinspection PyCallingNonCallable
            y = self(x)
            if isinstance(y, tuple):
                shape = tuple([get_out_shape(y[i]) for i in range(len(y))])
            else:
                shape = get_out_shape(y)
            return shape
        else:
            return -1

    @property
    def flat_shape(self):
        shape = self.shape
        try:
            return reduce(mul, shape)
        except TypeError:
            return shape


class F_x(ShapeMixin, nn.Module):
    def __init__(self, in_shape):
        super(F_x, self).__init__()
        self.in_shape = in_shape

    @staticmethod
    def forward(x):
        return x


class SlidingWindow(ShapeMixin, nn.Module):
    def __init__(self, in_shape, kernel, stride=1, padding=0, keepdim=False):
        super(SlidingWindow, self).__init__()
        self.in_shape = in_shape
        self.kernel = kernel if not isinstance(kernel, int) else (kernel, kernel)
        self.padding = padding
        self.stride = stride
        self.keepdim = keepdim
        self._unfolder = Unfold(self.kernel, dilation=1, padding=self.padding, stride=self.stride)

    def forward(self, x):
        tensor = self._unfolder(x)
        tensor = tensor.transpose(-1, -2)
        if self.keepdim:
            shape = *x.shape[:2], -1, *self.kernel
            tensor = tensor.reshape(shape)
        return tensor


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


class Filter(nn.Module, ShapeMixin):

    def __init__(self, in_shape, pos, dim=-1):
        super(Filter, self).__init__()

        self.in_shape = in_shape
        self.pos = pos
        self.dim = dim
        raise SystemError('Do not use this Module - broken.')

    @staticmethod
    def forward(x):
        tensor = x[:, -1]
        return tensor


class FlipTensor(nn.Module):
    def __init__(self, dim=-2):
        super(FlipTensor, self).__init__()
        self.dim = dim

    def forward(self, x):
        idx = [i for i in range(x.size(self.dim) - 1, -1, -1)]
        idx = torch.as_tensor(idx).long()
        inverted_tensor = x.index_select(self.dim, idx)
        return inverted_tensor


class AutoPadToShape(nn.Module):
    def __init__(self, target_shape):
        super(AutoPadToShape, self).__init__()
        self.target_shape = target_shape

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.shape[-len(self.target_shape):] == self.target_shape or x.shape == self.target_shape:
            return x

        idx = [0] * (len(self.target_shape) * 2)
        for i, j in zip(range(-1, -(len(self.target_shape)+1), -1), range(0, len(idx), 2)):
            idx[j] = self.target_shape[i] - x.shape[i]
        x = torch.nn.functional.pad(x, idx)
        return x

    def __repr__(self):
        return f'AutoPadTransform({self.target_shape})'


class Splitter(nn.Module):

    @property
    def shape(self):
        return tuple([self._out_shape] * self.n)

    @property
    def out_shape(self):
        return self._out_shape

    def __init__(self, in_shape, n, dim=-1):
        super(Splitter, self).__init__()

        self.in_shape = (in_shape, ) if isinstance(in_shape, int) else in_shape
        self.n = n
        self.dim = dim if dim > 0 else len(self.in_shape) - abs(dim)

        self.new_dim_size = (self.in_shape[self.dim] // self.n) + (1 if self.in_shape[self.dim] % self.n != 0 else 0)
        self._out_shape = tuple([x if self.dim != i else self.new_dim_size for i, x in enumerate(self.in_shape)])

        self.autopad = AutoPadToShape(self._out_shape)

    def forward(self, x: torch.Tensor):
        dim = self.dim + 1 if len(self.in_shape) == (x.ndim - 1) else self.dim
        x = x.transpose(0, dim)
        n_blocks = list()
        for block_idx in range(self.n):
            start = block_idx * self.new_dim_size
            end = (block_idx + 1) * self.new_dim_size
            block = x[start:end].transpose(0, dim)
            block = self.autopad(block)
            n_blocks.append(block)
        return n_blocks


class Merger(nn.Module, ShapeMixin):

    @property
    def shape(self):
        y = self.forward([torch.randn(self.in_shape) for _ in range(self.n)])
        return y.shape

    def __init__(self, in_shape, n, dim=-1):
        super(Merger, self).__init__()

        self.n = n
        self.dim = dim
        self.in_shape = in_shape

    def forward(self, x):
        return torch.cat(x, dim=self.dim)
