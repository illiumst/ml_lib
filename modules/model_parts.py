#
# Full Model Parts
###################
import torch
from torch import nn

from .util import ShapeMixin


class Generator(nn.Module):
    @property
    def shape(self):
        x = torch.randn(self.lat_dim).unsqueeze(0)
        output = self(x)
        return output.shape[1:]

    # noinspection PyUnresolvedReferences
    def __init__(self, out_channels, re_shape, lat_dim, use_norm=False, use_bias=True, dropout: Union[int, float] = 0,
                 filters: List[int] = None, activation=nn.ReLU):
        super(Generator, self).__init__()
        assert filters, '"Filters" has to be a list of int len 3'
        self.filters = filters
        self.activation = activation
        self.inner_activation = activation()
        self.out_activation = None
        self.lat_dim = lat_dim
        self.dropout = dropout
        self.l1 = nn.Linear(self.lat_dim, reduce(mul, re_shape), bias=use_bias)
        # re_shape = (self.feature_mixed_dim // reduce(mul, re_shape[1:]), ) + tuple(re_shape[1:])

        self.flat = Flatten(to=re_shape)

        self.deconv1 = DeConvModule(re_shape, conv_filters=self.filters[0],
                                    conv_kernel=5,
                                    conv_padding=2,
                                    conv_stride=1,
                                    normalize=use_norm,
                                    activation=self.activation,
                                    interpolation_scale=2,
                                    dropout=self.dropout
                                    )

        self.deconv2 = DeConvModule(self.deconv1.shape, conv_filters=self.filters[1],
                                    conv_kernel=3,
                                    conv_padding=1,
                                    conv_stride=1,
                                    normalize=use_norm,
                                    activation=self.activation,
                                    interpolation_scale=2,
                                    dropout=self.dropout
                                    )

        self.deconv3 = DeConvModule(self.deconv2.shape, conv_filters=self.filters[2],
                                    conv_kernel=3,
                                    conv_padding=1,
                                    conv_stride=1,
                                    normalize=use_norm,
                                    activation=self.activation,
                                    interpolation_scale=2,
                                    dropout=self.dropout
                                    )

        self.deconv4 = DeConvModule(self.deconv3.shape, conv_filters=out_channels,
                                    conv_kernel=3,
                                    conv_padding=1,
                                    # normalize=norm,
                                    activation=self.out_activation
                                    )

    def forward(self, z):
        tensor = self.l1(z)
        tensor = self.inner_activation(tensor)
        tensor = self.flat(tensor)
        tensor = self.deconv1(tensor)
        tensor = self.deconv2(tensor)
        tensor = self.deconv3(tensor)
        tensor = self.deconv4(tensor)
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


class BaseEncoder(ShapeMixin, nn.Module):

    # noinspection PyUnresolvedReferences
    def __init__(self, in_shape, lat_dim=256, use_bias=True, use_norm=False, dropout: Union[int, float] = 0,
                 latent_activation: Union[nn.Module, None] = None, activation: nn.Module = nn.ELU,
                 filters: List[int] = None):
        super(BaseEncoder, self).__init__()
        assert filters, '"Filters" has to be a list of int len 3'

        # Optional Padding for odd image-sizes
        # Obsolet, already Done by autopadding module on incoming tensors
        # in_shape = [x+1 if x % 2 != 0 and idx else x for idx, x in enumerate(in_shape)]

        # Parameters
        self.lat_dim = lat_dim
        self.in_shape = in_shape
        self.use_bias = use_bias
        self.latent_activation = latent_activation() if latent_activation else None

        # Modules
        self.conv1 = ConvModule(self.in_shape, conv_filters=filters[0],
                                conv_kernel=3,
                                conv_padding=1,
                                conv_stride=1,
                                pooling_size=2,
                                use_norm=use_norm,
                                dropout=dropout,
                                activation=activation
                                )

        self.conv2 = ConvModule(self.conv1.shape, conv_filters=filters[1],
                                conv_kernel=3,
                                conv_padding=1,
                                conv_stride=1,
                                pooling_size=2,
                                use_norm=use_norm,
                                dropout=dropout,
                                activation=activation
                                )

        self.conv3 = ConvModule(self.conv2.shape, conv_filters=filters[2],
                                conv_kernel=5,
                                conv_padding=2,
                                conv_stride=1,
                                pooling_size=2,
                                use_norm=use_norm,
                                dropout=dropout,
                                activation=activation
                                )

        self.flat = Flatten()

    def forward(self, x):
        tensor = self.conv1(x)
        tensor = self.conv2(tensor)
        tensor = self.conv3(tensor)
        tensor = self.flat(tensor)
        return tensor


class UnitEncoder(BaseEncoder):
    # noinspection PyUnresolvedReferences
    def __init__(self, *args, **kwargs):
        kwargs.update(use_norm=True)
        super(UnitEncoder, self).__init__(*args, **kwargs)
        self.l1 = nn.Linear(reduce(mul, self.conv3.shape), self.lat_dim, bias=self.use_bias)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        tensor = self.flat(c3)
        l1 = self.l1(tensor)
        return c1, c2, c3, l1


class VariationalEncoder(BaseEncoder):
    # noinspection PyUnresolvedReferences
    def __init__(self, *args, **kwargs):
        super(VariationalEncoder, self).__init__(*args, **kwargs)

        self.logvar = nn.Linear(reduce(mul, self.conv3.shape), self.lat_dim, bias=self.use_bias)
        self.mu = nn.Linear(reduce(mul, self.conv3.shape), self.lat_dim, bias=self.use_bias)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        tensor = super(VariationalEncoder, self).forward(x)
        mu = self.mu(tensor)
        logvar = self.logvar(tensor)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z


class Encoder(BaseEncoder):
    # noinspection PyUnresolvedReferences
    def __init__(self, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)

        self.l1 = nn.Linear(reduce(mul, self.conv3.shape), self.lat_dim, bias=self.use_bias)

    def forward(self, x):
        tensor = super(Encoder, self).forward(x)
        tensor = self.l1(tensor)
        tensor = self.latent_activation(tensor) if self.latent_activation else tensor
        return tensor
