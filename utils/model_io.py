from argparse import Namespace
from collections import Mapping
from pathlib import Path

import torch
from natsort import natsorted
from torch import nn


# Hyperparamter Object
class ModelParameters(Mapping, Namespace):

    @property
    def module_paramters(self):
        paramter_mapping = dict()
        paramter_mapping.update(self.model_param.__dict__)

        paramter_mapping.update(
            dict(
                activation=self._activations[paramter_mapping['activation']]
            )
        )

        del paramter_mapping['in_shape']

        return paramter_mapping

    def __getitem__(self, k):
        # k: _KT -> _VT_co
        return self.__dict__[k]

    def __len__(self):
        # -> int
        return len(self.__dict__.keys())

    def __iter__(self):
        # -> Iterator[_T_co]
        return iter(list(self.__dict__.keys()))

    def __delitem__(self, key):
        self.__dict__.__delitem__(key)
        return True

    _activations = dict(
        leaky_relu=nn.LeakyReLU,
        relu=nn.ReLU,
        sigmoid=nn.Sigmoid,
        tanh=nn.Tanh
    )

    def __init__(self, model_param, train_param, data_param):
        self.model_param = model_param
        self.train_param = train_param
        self.data_param = data_param
        kwargs = vars(model_param)
        kwargs.update(vars(train_param))
        kwargs.update(vars(data_param))
        super(ModelParameters, self).__init__(**kwargs)

    def __getattribute__(self, item):
        if item == 'activation':
            try:
                return self._activations[item]
            except KeyError:
                return nn.ReLU
        return super(ModelParameters, self).__getattribute__(item)


class SavedLightningModels(object):

    @classmethod
    def load_checkpoint(cls, models_root_path, model=None, n=-1, tags_file_path=''):
        assert models_root_path.exists(), f'The path {models_root_path.absolute()} does not exist!'
        found_checkpoints = list(Path(models_root_path).rglob('*.ckpt'))

        found_checkpoints = natsorted(found_checkpoints, key=lambda y: y.name)
        if model is None:
            model = torch.load(models_root_path / 'model_class.obj')
        assert model is not None

        return cls(weights=found_checkpoints[n], model=model)

    def __init__(self, **kwargs):
        self.weights: str = kwargs.get('weights', '')

        self.model = kwargs.get('model', None)
        assert self.model is not None

    def restore(self):
        pretrained_model = self.model.load_from_checkpoint(self.weights)
        pretrained_model.eval()
        pretrained_model.freeze()
        return pretrained_model