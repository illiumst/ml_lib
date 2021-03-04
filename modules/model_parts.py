#
# Full Model Parts
###################
from argparse import Namespace
from functools import reduce
from typing import Union, List, Tuple

import torch
from abc import ABC
from operator import mul
from torch import nn
from torch.utils.data import DataLoader

from .blocks import ConvModule, DeConvModule, LinearModule

from .util import ShapeMixin, LightningBaseModule, Flatten


class AEBaseModule(LightningBaseModule, ABC):

    def generate_random_image(self, dataloader: Union[None, str, DataLoader] = None,
                              lat_min: Union[Tuple, List, None] = None,
                              lat_max: Union[Tuple, List, None] = None):

        assert bool(dataloader) ^ bool(lat_min and lat_max), 'Decide wether to give min, max or a dataloader, not both.'

        min_max = self._find_min_max(dataloader) if dataloader else [None, None]
        # assert not any([tensor is None for tensor in min_max])
        lat_min = torch.as_tensor(lat_min or min_max[0])
        lat_max = lat_max or min_max[1]

        random_z = torch.rand((1, self.lat_dim))
        random_z = random_z * (abs(lat_min) + lat_max) - abs(lat_min)

        return self.decoder(random_z).squeeze()

    def encode(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.encoder(x)

    def _find_min_max(self, dataloader):
        encodings = list()
        for batch in dataloader:
            encodings.append(self.encode(batch))
        encodings = torch.cat(encodings, dim=0)
        min_lat = encodings.min(dim=1)
        max_lat = encodings.max(dim=1)
        return min_lat, max_lat

    def decode_lat_evenly(self, n: int,
                          dataloader: Union[None, str, DataLoader] = None,
                          lat_min: Union[Tuple, List, None] = None,
                          lat_max: Union[Tuple, List, None] = None):
        assert bool(dataloader) ^ bool(lat_min and lat_max), 'Decide wether to give min, max or a dataloader, not both.'

        min_max = self._find_min_max(dataloader) if dataloader else [None, None]

        lat_min = lat_min or min_max[0]
        lat_max = lat_max or min_max[1]

        random_latent_samples = torch.stack([torch.linspace(lat_min[i].item(), lat_max[i].item(), n)
                                             for i in range(self.params.lat_dim)], dim=-1).cpu().detach()
        return self.decode(random_latent_samples).cpu().detach()

    def decode(self, z):
        try:
            if len(z.shape) == 1:
                z = z.unsqueeze(0)
        except AttributeError:
            # Does not seem to be a tensor.
            pass
        return self.decoder(z).squeeze()

    def encode_and_restore(self, x):
        x = self.transfer_batch_to_device(x, self.device)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        z = self.encode(x)
        try:
            z = z.squeeze()
        except AttributeError:
            # Does not seem to be a tensor.
            pass
        x_hat = self.decode(z)

        return Namespace(main_out=x_hat.squeeze(), latent_out=z)


class Generator(ShapeMixin, nn.Module):

    def __init__(self, in_shape, out_channels, re_shape, use_norm=False, use_bias=True,
                 dropout: Union[int, float] = 0, interpolations: List[int] = None,
                 filters: List[int] = None, kernels: List[int] = None, activation=nn.ReLU,
                 **kwargs):
        super(Generator, self).__init__()
        assert filters, '"Filters" has to be a list of int.'
        assert filters, '"Filters" has to be a list of int.'
        kernels = kernels if kernels else [3] * len(filters)
        assert len(filters) == len(kernels), '"Filters" and "Kernels" has to be of same length.'

        interpolations = interpolations or [2, 2, 2]

        self.in_shape = in_shape
        self.activation = activation()
        self.out_activation = None
        self.dropout = dropout
        self.l1 = LinearModule(in_shape, reduce(mul, re_shape), bias=use_bias, activation=activation)
        # re_shape = (self.feature_mixed_dim // reduce(mul, re_shape[1:]), ) + tuple(re_shape[1:])

        self.flat = Flatten(self.l1.shape, to=re_shape)
        self.de_conv_list = nn.ModuleList()

        last_shape = re_shape
        for conv_filter, conv_kernel, interpolation in zip(reversed(filters), kernels, interpolations):
            # noinspection PyTypeChecker
            self.de_conv_list.append(DeConvModule(last_shape, conv_filters=conv_filter,
                                                  conv_kernel=conv_kernel,
                                                  conv_padding=conv_kernel-2,
                                                  conv_stride=1,
                                                  normalize=use_norm,
                                                  activation=activation,
                                                  interpolation_scale=interpolation,
                                                  dropout=self.dropout
                                                  )
                                     )
            last_shape = self.de_conv_list[-1].shape

        self.de_conv_out = DeConvModule(self.de_conv_list[-1].shape, conv_filters=out_channels, conv_kernel=3,
                                        conv_padding=1, activation=self.out_activation
                                        )

    def forward(self, z):
        tensor = self.l1(z)
        tensor = self.activation(tensor)
        tensor = self.flat(tensor)

        for de_conv in self.de_conv_list:
            tensor = de_conv(tensor)

        tensor = self.de_conv_out(tensor)
        return tensor

    def size(self):
        return self.shape


class UnitGenerator(Generator):

    def __init__(self, *args, **kwargs):
        kwargs.update(use_norm=True)
        super(UnitGenerator, self).__init__(*args, **kwargs)
        self.norm_f = nn.BatchNorm1d(self.l1.out_features, eps=1e-04, affine=False)
        self.norm1 = nn.BatchNorm2d(self.deconv1.conv_filters, eps=1e-04, affine=False)
        self.norm2 = nn.BatchNorm2d(self.deconv2.conv_filters, eps=1e-04, affine=False)
        self.norm3 = nn.BatchNorm2d(self.deconv3.conv_filters, eps=1e-04, affine=False)

    def forward(self, z_c1_c2_c3):
        z, c1, c2, c3 = z_c1_c2_c3
        tensor = self.l1(z)
        tensor = self.inner_activation(tensor)
        tensor = self.norm(tensor)
        tensor = self.flat(tensor)

        tensor = self.deconv1(tensor) + c3
        tensor = self.inner_activation(tensor)
        tensor = self.norm1(tensor)

        tensor = self.deconv2(tensor) + c2
        tensor = self.inner_activation(tensor)
        tensor = self.norm2(tensor)

        tensor = self.deconv3(tensor) + c1
        tensor = self.inner_activation(tensor)
        tensor = self.norm3(tensor)

        tensor = self.deconv4(tensor)
        return tensor


class BaseCNNEncoder(ShapeMixin, nn.Module):

    # noinspection PyUnresolvedReferences
    def __init__(self, in_shape, lat_dim=256, use_bias=True, use_norm=False, dropout: Union[int, float] = 0,
                 latent_activation: Union[nn.Module, None] = None, activation: nn.Module = nn.ELU,
                 filters: List[int] = None, kernels: Union[List[int], int, None] = None, **kwargs):
        super(BaseCNNEncoder, self).__init__()
        assert filters, '"Filters" has to be a list of int'
        kernels = kernels or [3] * len(filters)
        kernels = kernels if not isinstance(kernels, int) else [kernels] * len(filters)
        assert len(kernels) == len(filters), 'Length of "Filters" and "Kernels" has to be same.'

        # Optional Padding for odd image-sizes
        # Obsolet, cdan be done by autopadding module on incoming tensors
        # in_shape = [tensor+1 if tensor % 2 != 0 and idx else tensor for idx, tensor in enumerate(in_shape)]

        # Parameters
        self.lat_dim = lat_dim
        self.in_shape = in_shape
        self.use_bias = use_bias
        self.latent_activation = latent_activation() if latent_activation else None

        self.conv_list = nn.ModuleList()

        # Modules
        last_shape = self.in_shape
        for conv_filter, conv_kernel in zip(filters, kernels):
            self.conv_list.append(ConvModule(last_shape, conv_filters=conv_filter,
                                             conv_kernel=conv_kernel,
                                             conv_padding=conv_kernel-2,
                                             conv_stride=1,
                                             pooling_size=2,
                                             use_norm=use_norm,
                                             dropout=dropout,
                                             activation=activation
                                             )
                                  )
            last_shape = self.conv_list[-1].shape
        self.last_conv_shape = last_shape

        self.flat = Flatten(self.last_conv_shape)

    def forward(self, x):
        tensor = x
        for conv in self.conv_list:
            tensor = conv(tensor)
        tensor = self.flat(tensor)
        return tensor


class UnitCNNEncoder(BaseCNNEncoder):
    # noinspection PyUnresolvedReferences
    def __init__(self, *args, **kwargs):
        kwargs.update(use_norm=True)
        super(UnitCNNEncoder, self).__init__(*args, **kwargs)
        self.l1 = nn.Linear(reduce(mul, self.conv3.shape), self.lat_dim, bias=self.use_bias)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        tensor = self.flat(c3)
        l1 = self.l1(tensor)
        return c1, c2, c3, l1


class VariationalCNNEncoder(BaseCNNEncoder):
    # noinspection PyUnresolvedReferences
    def __init__(self, *args, **kwargs):
        super(VariationalCNNEncoder, self).__init__(*args, **kwargs)

        self.logvar = nn.Linear(reduce(mul, self.conv3.shape), self.lat_dim, bias=self.use_bias)
        self.mu = nn.Linear(reduce(mul, self.conv3.shape), self.lat_dim, bias=self.use_bias)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        tensor = super(VariationalCNNEncoder, self).forward(x)
        mu = self.mu(tensor)
        logvar = self.logvar(tensor)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z


class CNNEncoder(BaseCNNEncoder):

    def __init__(self, *args, **kwargs):
        super(CNNEncoder, self).__init__(*args, **kwargs)

        self.l1 = nn.Linear(self.flat.shape, self.lat_dim, bias=self.use_bias)

    def forward(self, x):
        tensor = super(CNNEncoder, self).forward(x)
        tensor = self.l1(tensor)
        tensor = self.latent_activation(tensor) if self.latent_activation else tensor
        return tensor
