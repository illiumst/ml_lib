import warnings

from pathlib import Path
from typing import Union

import torch

from torch import nn
from torch.nn import functional as F

from einops import rearrange, repeat

import sys
sys.path.append(str(Path(__file__).parent))

from .util import AutoPad, Interpolate, ShapeMixin, F_x, Flatten

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#
# Sub - Modules
###################
class LinearModule(ShapeMixin, nn.Module):

    def __init__(self, in_shape, out_features, use_bias=True, activation=None,
                 use_norm=False, dropout: Union[int, float] = 0, **kwargs):
        if list(kwargs.keys()):
            warnings.warn(f'The following arguments have been ignored: \n {list(kwargs.keys())}')
        super(LinearModule, self).__init__()

        self.in_shape = in_shape
        self.flat = Flatten(self.in_shape) if isinstance(self.in_shape, (tuple, list)) else F_x(in_shape)
        self.dropout = nn.Dropout(dropout) if dropout else F_x(self.flat.shape)
        self.norm = nn.LayerNorm(self.flat.shape) if use_norm else F_x(self.flat.shape)
        self.linear = nn.Linear(self.flat.shape, out_features, bias=use_bias)
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
                 bias=True, use_norm=False, dropout: Union[int, float] = 0, trainable: bool = True,
                 conv_class=nn.Conv2d, conv_stride=1, conv_padding=0, **kwargs):
        super(ConvModule, self).__init__()
        assert isinstance(in_shape, (tuple, list)), f'"in_shape" should be a [list, tuple], but was {type(in_shape)}'
        assert len(in_shape) == 3, f'Length should be 3, but was {len(in_shape)}'
        if len(kwargs.keys()):
            warnings.warn(f'The following arguments have been ignored: \n {list(kwargs.keys())}')
        if use_norm and not trainable:
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
        self.activation = activation() or nn.Identity()
        self.norm = nn.LayerNorm(self.in_shape, eps=1e-04) if use_norm else F_x(None)
        self.dropout = nn.Dropout2d(dropout) if dropout else F_x(None)
        self.pooling = nn.MaxPool2d(pooling_size) if pooling_size else F_x(None)
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
        self.weight_matrix = weight_matrix
        raise NotImplementedError
        # ToDo Get the weight_matrix shape and init a conv_module of similar size,
        #      override the weights then.

    def forward(self, x):
        x = torch.matmul(x, self.weight_matrix)  # ToDo: This is an Placeholder
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
                 bias=True, use_norm=False, **kwargs):
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
        self.norm = nn.LayerNorm(in_channels, eps=1e-04) if use_norm else F_x(self.in_shape)
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

    def __init__(self, in_shape, module_class, n, use_norm=False, **module_parameters):
        assert n >= 1
        super(ResidualModule, self).__init__()
        self.in_shape = in_shape
        module_parameters.update(in_shape=in_shape)
        if use_norm:
            self.norm = nn.LayerNorm(self.in_shape if isinstance(self.in_shape, int) else self.in_shape[0])
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


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., activation=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            activation() or F_x(None),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            activation() or F_x(None),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, head_dim=64, dropout=0.):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None, return_attn_weights=False):
        # noinspection PyTupleAssignmentBalance
        b, n, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            mask = repeat(mask, 'b n d -> b h n d', h=h)             # My addition

            #dots.masked_fill_(~mask, mask_value)
            dots.masked_fill_(mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if return_attn_weights:
            return out, attn
        else:
            return out


class TransformerModule(ShapeMixin, nn.Module):

    def __init__(self, in_shape, depth, heads, mlp_dim, head_dim=32, dropout=None, use_norm=False,
                 activation=nn.GELU, use_residual=True):
        super(TransformerModule, self).__init__()

        self.in_shape = in_shape
        self.use_residual = use_residual

        self.flat = Flatten(self.in_shape) if isinstance(self.in_shape, (tuple, list)) else F_x(in_shape)

        self.embedding_dim = self.flat.flat_shape
        self.norm = nn.LayerNorm(self.embedding_dim) if use_norm else F_x(None)
        self.attns = nn.ModuleList([Attention(self.embedding_dim, heads=heads, dropout=dropout, head_dim=head_dim)
                                    for _ in range(depth)])
        self.mlps = nn.ModuleList([FeedForward(self.embedding_dim, mlp_dim, dropout=dropout, activation=activation)
                                   for _ in range(depth)])

    def forward(self, x, mask=None, return_attn_weights=False, **_):
        tensor = self.flat(x)
        attn_weights = list()

        for attn, mlp in zip(self.attns, self.mlps):
            # Attention
            attn_tensor = self.norm(tensor)
            if return_attn_weights:
                attn_tensor, attn_weight = attn(attn_tensor, mask=mask, return_attn_weights=return_attn_weights)
                attn_weights.append(attn_weight)
            else:
                attn_tensor = attn(attn_tensor, mask=mask)
            tensor = tensor + attn_tensor if self.use_residual else attn_tensor

            # MLP
            mlp_tensor = self.norm(tensor)
            mlp_tensor = mlp(mlp_tensor)
            tensor = tensor + mlp_tensor if self.use_residual else mlp_tensor

        return (tensor, attn_weights) if return_attn_weights else tensor
