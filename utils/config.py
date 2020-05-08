import ast
from copy import deepcopy

from abc import ABC

from argparse import Namespace, ArgumentParser
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path
import hashlib


def is_jsonable(x):
    import json
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False


class Config(ConfigParser, ABC):

    @property
    def name(self):
        short_name = "".join(c for c in self.model.type if c.isupper())
        return f'{short_name}_{self.fingerprint}'

    @property
    def version(self):
        return f'version_{self.main.seed}'

    @property
    def exp_path(self):
        return Path(self.train.outpath) / self.model.type / self.name

    @property
    def fingerprint(self):
        h = hashlib.md5()
        params = deepcopy(self.as_dict)
        del params['model']['type']
        del params['data']['worker']
        del params['main']
        h.update(str(params).encode())
        fingerprint = h.hexdigest()
        return fingerprint

    @property
    def _model_weight_init(self):
        mod = __import__('torch.nn.init', fromlist=[self.model.weight_init])
        return getattr(mod, self.model.weight_init)

    @property
    def _model_map(self):
        """
        This is function is supposed to return a dict, which holds a mapping from string model names to model classes

        Example:
        from models.binary_classifier import ConvClassifier
        return dict(ConvClassifier=ConvClassifier,
                    )
        :return:
        """

        raise NotImplementedError

    @property
    def model_class(self):
        try:
            return self._model_map[self.model.type]
        except KeyError:
            raise KeyError(rf'The model alias you provided ("{self.get("model", "type")}") does not exist! Try one of these: {list(self._model_map.keys())}')

    # TODO: Do this programmatically; This did not work:
    # Initialize Default Sections as Property
    # for section in self.default_sections:
    #     self.__setattr__(section, property(lambda x :x._get_namespace_for_section(section))

    @property
    def main(self):
        return self._get_namespace_for_section('main')

    @property
    def model(self):
        return self._get_namespace_for_section('model')

    @property
    def train(self):
        return self._get_namespace_for_section('train')

    @property
    def data(self):
        return self._get_namespace_for_section('data')

    @property
    def project(self):
        return self._get_namespace_for_section('project')

    ###################################################

    @property
    def model_paramters(self):
        params = deepcopy(self.model.__dict__)
        assert all(key not in list(params.keys()) for key in self.train.__dict__)
        params.update(self.train.__dict__)
        assert all(key not in list(params.keys()) for key in self.data.__dict__)
        params.update(self.data.__dict__)
        params.update(exp_path=str(self.exp_path), exp_fingerprint=str(self.fingerprint))
        return params

    @property
    def tags(self, ):
        return [f'{key}: {val}' for key, val in self.serializable.items()]

    @property
    def serializable(self):
        return {f'{section}_{key}': val for section, params in self._sections.items()
                for key, val in params.items() if is_jsonable(val)}

    @property
    def as_dict(self):
        return self._sections

    def _get_namespace_for_section(self, item):
        return Namespace(**{key: self.get(item, key) for key in self[item]})

    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)
        pass

    @staticmethod
    def _sort_combined_section_key_mapping(dict_obj):
        sorted_dict = defaultdict(dict)
        for key in dict_obj:
            section, *attr_name = key.split('_')
            attr_name = '_'.join(attr_name)
            value = str(dict_obj[key])

            sorted_dict[section][attr_name] = value
        # noinspection PyTypeChecker
        return dict(sorted_dict)

    @classmethod
    def read_namespace(cls, namespace: Namespace):

        sorted_dict = cls._sort_combined_section_key_mapping(namespace.__dict__)
        new_config = cls()
        new_config.read_dict(sorted_dict)
        return new_config

    @classmethod
    def read_argparser(cls, argparser: ArgumentParser):
        # Parse it
        args = argparser.parse_args()
        sorted_dict = cls._sort_combined_section_key_mapping(args.__dict__)
        new_config = cls()
        new_config.read_dict(sorted_dict)
        return new_config

    def build_model(self):
        return self.model_class(self.model_paramters)

    def build_and_init_model(self):
        model = self.build_model()
        model.init_weights(self._model_weight_init)
        return model

    def update(self, mapping):
        sorted_dict = self._sort_combined_section_key_mapping(mapping)
        for section in sorted_dict:
            if self.has_section(section):
                pass
            else:
                self.add_section(section)
            for option, value in sorted_dict[section].items():
                self.set(section, option, value)
        return self

    def get(self, *args, **kwargs):
        item = super(Config, self).get(*args, **kwargs)
        try:
            return ast.literal_eval(item)
        except SyntaxError:
            return item
        except ValueError:
            return item

    def write(self, filepath, **kwargs):
        path = Path(filepath, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('w') as configfile:
            super().write(configfile)
        return True
