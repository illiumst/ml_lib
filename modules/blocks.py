import math

from pathlib import Path
from typing import Union

import torch
import warnings

from torch import nn
from torch.nn import functional as F
import sys
sys.path.append(str(Path(__file__).parent))
from .util import AutoPad, Interpolate, ShapeMixin, F_x, Flatten

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#
# Sub - Modules
###################
class LinearModule(ShapeMixin, nn.Module):

    def __init__(self, in_shape, out_features, bias=True, activation=None,
                 norm=False, dropout: Union[int, float] = 0, **kwargs):
        if list(kwargs.keys()):
            warnings.warn(f'The following arguments have been ignored: \n {list(kwargs.keys())}')
        super(LinearModule, self).__init__()

        self.in_shape = in_shape
        self.flat = Flatten(self.in_shape) if isinstance(self.in_shape, (tuple, list)) else F_x(in_shape)
        self.dropout = nn.Dropout(dropout) if dropout else F_x(self.flat.shape)
        self.norm = nn.BatchNorm1d(self.flat.shape) if norm else F_x(self.flat.shape)
        self.linear = nn.Linear(self.flat.shape, out_features, bias=bias)
        self.activation = activation() if activation else F_x(self.linear.out_features)

    def forward(self, x):
        tensor = self.flat(x)
        tensor = self.dropout(tensor)
        tensor = self.norm(tensor)
        tensor = self.linear(tensor)
        tensor = self.activation(tensor)
        return tensor


class ConvModule(ShapeMixin, nn.Module):

    def __init__(self, in_shape, conv_filters, conv_kernel, activation: nn.Module = nn.ELU, pooling_size=None,
                 bias=True, norm=False, dropout: Union[int, float] = 0, trainable: bool = True,
                 conv_class=nn.Conv2d, conv_stride=1, conv_padding=0, **kwargs):
        super(ConvModule, self).__init__()
        assert isinstance(in_shape, (tuple, list)), f'"in_shape" should be a [list, tuple], but was {type(in_shape)}'
        assert len(in_shape) == 3, f'Length should be 3, but was {len(in_shape)}'
        warnings.warn(f'The following arguments have been ignored: \n {list(kwargs.keys())}')
        if norm and not trainable:
            warnings.warn('You set this module to be not trainable but the running norm is active.\n' +
                          'We set it to "eval" mode.\n' +
                          'Keep this in mind if you do a finetunning or retraining step.'
                          )

        # Module Parameters
        self.in_shape = in_shape
        self.trainable = trainable
        in_channels, height, width = in_shape[0], in_shape[1], in_shape[2]

        # Convolution Parameters
        self.padding = conv_padding
        self.stride = conv_stride
        self.conv_filters = conv_filters
        self.conv_kernel = conv_kernel

        # Modules
        self.activation = activation() or F_x(None)
        self.dropout = nn.Dropout2d(dropout) if dropout else F_x(None)
        self.pooling = nn.MaxPool2d(pooling_size) if pooling_size else F_x(None)
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-04) if norm else F_x(None)
        self.conv = conv_class(in_channels, self.conv_filters, self.conv_kernel, bias=bias,
                               padding=self.padding, stride=self.stride
                               )
        if not self.trainable:
            for param in self.parameters():
                param.requires_grad = False
            self.norm = self.norm.eval()
        else:
            pass


    def forward(self, x):
        tensor = self.norm(x)
        tensor = self.conv(tensor)
        tensor = self.dropout(tensor)
        tensor = self.pooling(tensor)
        tensor = self.activation(tensor)
        return tensor


class PreInitializedConvModule(ShapeMixin, nn.Module):

    def __init__(self, in_shape, weight_matrix):
        super(PreInitializedConvModule, self).__init__()
        self.in_shape = in_shape
        raise NotImplementedError
        # ToDo Get the weight_matrix shape and init a conv_module of similar size,
        #      override the weights then.

    def forward(self, x):

        return x


class SobelFilter(ShapeMixin, nn.Module):

    def __init__(self, in_shape):
        super(SobelFilter, self).__init__()
        self.in_shape = in_shape

        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, 2, -1]]).view(1, 1, 3, 3)

    def forward(self, x):
        # Apply Filters
        g_x = F.conv2d(x, self.sobel_x)
        g_y = F.conv2d(x, self.sobel_y)
        # Calculate the Edge
        g = torch.add(*[torch.pow(tensor, 2) for tensor in [g_x, g_y]])
        # Calculate the Gradient
        g_grad = torch.atan2(g_x, g_y)
        return g_x, g_y, g, g_grad


class DeConvModule(ShapeMixin, nn.Module):

    def __init__(self, in_shape, conv_filters, conv_kernel, conv_stride=1, conv_padding=0,
                 dropout: Union[int, float] = 0, autopad=0,
                 activation: Union[None, nn.Module] = nn.ReLU, interpolation_scale=0,
                 bias=True, norm=False, **kwargs):
        super(DeConvModule, self).__init__()
        warnings.warn(f'The following arguments have been ignored: \n {list(kwargs.keys())}')
        in_channels, height, width = in_shape[0], in_shape[1], in_shape[2]
        self.padding = conv_padding
        self.conv_kernel = conv_kernel
        self.stride = conv_stride
        self.in_shape = in_shape
        self.conv_filters = conv_filters

        self.autopad = AutoPad() if autopad else lambda x: x
        self.interpolation = Interpolate(scale_factor=interpolation_scale) if interpolation_scale else lambda x: x
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-04) if norm else F_x(self.in_shape)
        self.dropout = nn.Dropout2d(dropout) if dropout else F_x(self.in_shape)
        self.de_conv = nn.ConvTranspose2d(in_channels, self.conv_filters, self.conv_kernel, bias=bias,
                                          padding=self.padding, stride=self.stride)

        self.activation = activation() if activation else lambda x: x

    def forward(self, x):
        x = self.norm(x)
        x = self.dropout(x)
        x = self.autopad(x)
        x = self.interpolation(x)

        tensor = self.de_conv(x)
        tensor = self.activation(tensor)
        return tensor


class ResidualModule(ShapeMixin, nn.Module):

    def __init__(self, in_shape, module_class, n, norm=False, **module_parameters):
        assert n >= 1
        super(ResidualModule, self).__init__()
        self.in_shape = in_shape
        module_parameters.update(in_shape=in_shape)
        if norm:
            norm = nn.BatchNorm1d if len(self.in_shape) <= 2 else nn.BatchNorm2d
            self.norm = norm(self.in_shape if isinstance(self.in_shape, int) else self.in_shape[0])
        else:
            self.norm = F_x(self.in_shape)
        self.activation = module_parameters.get('activation', None)
        if self.activation is not None:
            self.activation = self.activation()
        else:
            self.activation = F_x(self.in_shape)
        self.residual_block = nn.ModuleList([module_class(**module_parameters) for _ in range(n)])
        assert self.in_shape == self.shape, f'The in_shape: {self.in_shape} - must match the out_shape: {self.shape}.'

    def forward(self, x):
        tensor = self.norm(x)
        for module in self.residual_block:
            tensor = module(tensor)

        # noinspection PyUnboundLocalVariable
        tensor = tensor + x
        tensor = self.activation(tensor)
        return tensor


class RecurrentModule(ShapeMixin, nn.Module):

    def __init__(self, in_shape, hidden_size, num_layers=1, cell_type=nn.GRU, bias=True, dropout=0):
        super(RecurrentModule, self).__init__()
        self.bias = bias
        self.num_layers = num_layers
        self.in_shape = in_shape
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.rnn = cell_type(self.in_shape[-1] * self.in_shape[-2], hidden_size,
                             num_layers=num_layers,
                             bias=self.bias,
                             batch_first=True,
                             dropout=self.dropout)

    def forward(self, x):
        tensor = self.rnn(x)
        return tensor


class AttentionModule(ShapeMixin, nn.Module):
    def __init__(self,in_shape, features, dropout=0.1):
        super().__init__()
        self.in_shape = in_shape
        self.dropout = dropout
        self.features = features
        raise NotImplementedError

    def forward(self, x):
        pass


class MultiHeadAttentionModule(ShapeMixin, nn.Module):
    def __init__(self, in_shape, heads, features, dropout=0.1):
        super().__init__()
        self.in_shape = in_shape

        self.features = features
        self.heads = heads
        self.final_dim = self.features // self.heads

        self.linear_q = LinearModule(self.features, self.features)
        self.linear_v = LinearModule(self.features, self.features)
        self.linear_k = LinearModule(self.features, self.features)
        self.dropout = nn.Dropout(dropout) if dropout else F_x(self.features)
        self.linear_out = nn.Linear(self.features, self.features)

    def forward(self, q, k, v, mask=None):

        batch_size = q.size(0)

        # perform linear operation and split into h heads
        k = self.linear_k(k).view(batch_size, -1, self.heads, self.final_dim)
        q = self.linear_q(q).view(batch_size, -1, self.heads, self.final_dim)
        v = self.linear_v(v).view(batch_size, -1, self.heads, self.final_dim)

        # transpose to get dimensions bs * h * sl * features
        # ToDo: Do we need this?

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.final_dim)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        scores = torch.matmul(scores, v)

        # concatenate heads and apply final linear transformation
        # ToDo: This seems to be old coding style. Do we Need this?
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.features)

        output = self.out(concat)
        return output


class TransformerModule(ShapeMixin, nn.Module):

    def __init__(self, in_shape, hidden_size, n_heads, num_layers=1, dropout=None, use_norm=False, **kwargs):
        super(TransformerModule, self).__init__()

        self.in_shape = in_shape

        self.flat = Flatten(self.in_shape) if isinstance(self.in_shape, (tuple, list)) else F_x(in_shape)

        encoder_layer = nn.TransformerEncoderLayer(self.flat_shape, n_heads, dim_feedforward=hidden_size,
                                                   dropout=dropout, activation=kwargs.get('activation')
                                                   )
        self.norm = nn.LayerNorm(hidden_size) if use_norm else F_x(hidden_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, )

    def forward(self, x, mask=None, key_padding_mask=None):
        tensor = self.flat(x)
        tensor = self.transformer(tensor, mask, key_padding_mask)
        return tensor
