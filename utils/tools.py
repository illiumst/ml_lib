import importlib
import inspect
import pickle
import shelve
from argparse import ArgumentParser, ArgumentError
from ast import literal_eval
from pathlib import Path, PurePath
from typing import Union

import numpy as np
import torch
import random


def auto_cast(a):
  try:
    return literal_eval(a)
  except:
    return a


def to_one_hot(idx_array, max_classes):
    one_hot = np.zeros((idx_array.size, max_classes))
    one_hot[np.arange(idx_array.size), idx_array] = 1
    return one_hot


def fix_all_random_seeds(config_obj):
    np.random.seed(config_obj.main.seed)
    torch.manual_seed(config_obj.main.seed)
    random.seed(config_obj.main.seed)


def write_to_shelve(file_path, value):
    check_path(file_path)
    file_path.parent.mkdir(exist_ok=True, parents=True)
    with shelve.open(str(file_path), protocol=pickle.HIGHEST_PROTOCOL) as f:
        new_key = str(len(f))
        f[new_key] = value
    f.close()


def load_from_shelve(file_path, key):
    check_path(file_path)
    with shelve.open(str(file_path)) as d:
        return d[key]


def check_path(file_path):
    assert isinstance(file_path, Path)
    assert str(file_path).endswith('.pik')


def locate_and_import_class(class_name, folder_path: Union[str, PurePath] = ''):
    """Locate an object by name or dotted path, importing as necessary."""
    folder_path = Path(folder_path)
    module_paths = [x for x in folder_path.rglob('*.py') if x.is_file() and '__init__' not in x.name]
    for module_path in module_paths:
        mod = importlib.import_module('.'.join([x.replace('.py', '') for x in module_path.parts]))
        try:
            model_class = mod.__getattribute__(class_name)
            return model_class
        except AttributeError:
           continue
    raise AttributeError(f'Check the Model name. Possible model files are:\n{[x.name for x in module_paths]}')


def add_argparse_args(cls, parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    full_arg_spec = inspect.getfullargspec(cls.__init__)
    n_non_defaults = len(full_arg_spec.args) - (len(full_arg_spec.defaults) if full_arg_spec.defaults else 0)
    for idx, argument in enumerate(full_arg_spec.args):
        try:
            if argument == 'self':
                continue
            if idx < n_non_defaults:
                parser.add_argument(f'--{argument}', type=int)
            else:
                argument_type = type(argument)
                parser.add_argument(f'--{argument}',
                                    type=argument_type,
                                    default=full_arg_spec.defaults[idx - n_non_defaults]
                                    )
        except ArgumentError:
            continue
    return parser
