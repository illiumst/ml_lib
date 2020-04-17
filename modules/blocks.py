from typing import Union
import warnings

import torch
from torch import nn
from ml_lib.modules.utils import AutoPad, Interpolate


#
# Sub - Modules
###################

class ConvModule(nn.Module):

    @property
    def shape(self):
        x = torch.randn(self.in_shape).unsqueeze(0)
        output = self(x)
        return output.shape[1:]

    def __init__(self, in_shape, conv_filters, conv_kernel, activation: nn.Module = nn.ELU, pooling_size=None,
                 use_bias=True, use_norm=False, dropout: Union[int, float] = 0,
                 conv_class=nn.Conv2d, conv_stride=1, conv_padding=0, **kwargs):
        super(ConvModule, self).__init__()
        warnings.warn(f'The following arguments have been ignored: \n {list(kwargs.keys())}')
        # Module Parameters
        self.in_shape = in_shape
        in_channels, height, width = in_shape[0], in_shape[1], in_shape[2]
        self.activation = activation()

        # Convolution Parameters
        self.padding = conv_padding
        self.stride = conv_stride
        self.conv_filters = conv_filters
        self.conv_kernel = conv_kernel

        # Modules
        self.dropout = nn.Dropout2d(dropout) if dropout else lambda x: x
        self.pooling = nn.MaxPool2d(pooling_size) if pooling_size else lambda x: x
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-04) if use_norm else lambda x: x
        self.conv = conv_class(in_channels, self.conv_filters, self.conv_kernel, bias=use_bias,
                               padding=self.padding, stride=self.stride
                               )

    def forward(self, x):
        x = self.norm(x)

        tensor = self.conv(x)
        tensor = self.dropout(tensor)
        tensor = self.pooling(tensor)
        tensor = self.activation(tensor)
        return tensor


class DeConvModule(nn.Module):

    @property
    def shape(self):
        x = torch.randn(self.in_shape).unsqueeze(0)
        output = self(x)
        return output.shape[1:]

    def __init__(self, in_shape, conv_filters, conv_kernel, conv_stride=1, conv_padding=0,
                 dropout: Union[int, float] = 0, autopad=0,
                 activation: Union[None, nn.Module] = nn.ReLU, interpolation_scale=0,
                 use_bias=True, use_norm=False):
        super(DeConvModule, self).__init__()
        in_channels, height, width = in_shape[0], in_shape[1], in_shape[2]
        self.padding = conv_padding
        self.conv_kernel = conv_kernel
        self.stride = conv_stride
        self.in_shape = in_shape
        self.conv_filters = conv_filters

        self.autopad = AutoPad() if autopad else lambda x: x
        self.interpolation = Interpolate(scale_factor=interpolation_scale) if interpolation_scale else lambda x: x
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-04) if use_norm else lambda x: x
        self.dropout = nn.Dropout2d(dropout) if dropout else lambda x: x
        self.de_conv = nn.ConvTranspose2d(in_channels, self.conv_filters, self.conv_kernel, bias=use_bias,
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


class ResidualModule(nn.Module):

    @property
    def shape(self):
        x = torch.randn(self.in_shape).unsqueeze(0)
        output = self(x)
        return output.shape[1:]

    def __init__(self, in_shape, module_class, n, activation=None, **module_parameters):
        assert n >= 1
        super(ResidualModule, self).__init__()
        self.in_shape = in_shape
        module_parameters.update(in_shape=in_shape)
        self.activation = activation() if activation else lambda x: x
        self.residual_block = nn.ModuleList([module_class(**module_parameters) for _ in range(n)])
        assert self.in_shape == self.shape, f'The in_shape: {self.in_shape} - must match the out_shape: {self.shape}.'

    def forward(self, x):
        for module in self.residual_block:
            tensor = module(x)

        # noinspection PyUnboundLocalVariable
        tensor = tensor + x
        tensor = self.activation(tensor)
        return tensor


class RecurrentModule(nn.Module):

    @property
    def shape(self):
        x = torch.randn(self.in_shape).unsqueeze(0)
        output = self(x)
        return output.shape[1:]

    def __init__(self, in_shape, hidden_size, num_layers=1, cell_type=nn.GRU, use_bias=True, dropout=0):
        super(RecurrentModule, self).__init__()
        self.use_bias = use_bias
        self.num_layers = num_layers
        self.in_shape = in_shape
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.rnn = cell_type(self.in_shape[-1] * self.in_shape[-2], hidden_size,
                             num_layers=num_layers,
                             bias=self.use_bias,
                             batch_first=True,
                             dropout=self.dropout)

    def forward(self, x):
        tensor = self.rnn(x)
        return tensor
