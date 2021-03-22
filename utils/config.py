import ast
import configparser
from pathlib import Path
from typing import Mapping, Dict

import torch
from copy import deepcopy

from abc import ABC

from argparse import Namespace, ArgumentParser
from collections import defaultdict
from configparser import ConfigParser, DuplicateSectionError
import hashlib
from pytorch_lightning import Trainer

from ml_lib.utils.loggers import Logger
from ml_lib.utils.tools import locate_and_import_class, auto_cast


# Argument Parser and default Values
# =============================================================================
def parse_comandline_args_add_defaults(filepath, overrides=None):

    # Parse Command Line
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--seed', type=str)

    # Load Defaults from _parameters.ini file
    config = configparser.ConfigParser()
    config.read(str(filepath))

    new_defaults = dict()
    for key in ['project', 'train', 'data']:
        defaults = config[key]
        new_defaults.update({key: auto_cast(val) for key, val in defaults.items()})

    if new_defaults['debug']:
        new_defaults.update(
            max_epochs=2,
            max_steps=2     # The seems to be the new "fast_dev_run"
        )

    args, _ = parser.parse_known_args()
    overrides = overrides or dict()
    default_data = overrides.get('data_name', None) or new_defaults['data_name']
    default_model = overrides.get('model_name', None) or new_defaults['model_name']
    default_seed = overrides.get('seed', None) or new_defaults['seed']

    data_name = args.__dict__.get('data_name', None) or default_data
    model_name = args.__dict__.get('model_name', None) or default_model
    found_seed = args.__dict__.get('seed', None) or default_seed

    new_defaults.update({key: auto_cast(val) for key, val in config[model_name].items()})

    found_data_class = locate_and_import_class(data_name, 'datasets')
    found_model_class = locate_and_import_class(model_name, 'models')

    for module in [Logger, Trainer, found_data_class, found_model_class]:
        parser = module.add_argparse_args(parser)

    args, _ = parser.parse_known_args(namespace=Namespace(**new_defaults))

    args = vars(args)
    args.update({key: auto_cast(val) for key, val in args.items()})
    args.update(gpus=[0] if torch.cuda.is_available() and not args['debug'] else None,
                row_log_interval=1000,  # TODO: Better Value / Setting
                log_save_interval=10000,  # TODO: Better Value / Setting
                auto_lr_find=not args['debug'],
                weights_summary='top',
                check_val_every_n_epoch=1 if args['debug'] else args.get('check_val_every_n_epoch', 1),
                )

    if overrides is not None and isinstance(overrides, (Mapping, Dict)):
        args.update(**overrides)
    return args, found_data_class, found_model_class, found_seed


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
        try:
            del params['model']['type']
        except KeyError:
            pass
        try:
            del params['data']['worker']
        except KeyError:
            pass
        try:
            del params['data']['refresh']
        except KeyError:
            pass
        try:
            del params['main']
        except KeyError:
            pass
        try:
            del params['project']
        except KeyError:
            pass
        # Flatten the dict of dicts
        for section in list(params.keys()):
            params.update({f'{section}_{key}': val for key, val in params[section].items()})
            del params[section]
        _, vals = zip(*sorted(params.items(), key=lambda tup: tup[0]))
        h.update(str(vals).encode())
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
            return locate_and_import_class(self.model.type, folder_path='models')
        except AttributeError as e:
            raise AttributeError(f'The model alias you provided ("{self.get("model", "type")}") ' +
                                 f'was not found!\n' +
                                 f'{e}')

    @property
    def data_class(self):
        try:
            return locate_and_import_class(self.data.class_name, folder_path='datasets')
        except AttributeError as e:
            raise AttributeError(f'The dataset alias you provided ("{self.get("data", "class_name")}") ' +
                                 f'was not found!\n' +
                                 f'{e}')

    # --------------------------------------------------
    # TODO: Do this programmatically; This did not work:
    # Initialize Default Sections as Property
    # for section in self.default_sections:
    #     self.__setattr__(section, property(lambda tensor :tensor._get_namespace_for_section(section))

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
        params.update(version=self.version)
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

    def _write_section(self, fp, section_name, section_items, delimiter):
        if section_name == 'project':
            return
        else:
            super(Config, self)._write_section(fp, section_name, section_items, delimiter)

    def add_section(self, section: str) -> None:
        try:
            super(Config, self).add_section(section)
        except DuplicateSectionError:
            pass


class DataClass(Namespace):

    @property
    def __dict__(self):
        return [x for x in dir(self) if not x.startswith('_')]
