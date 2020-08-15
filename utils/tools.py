import importlib
import pickle
import shelve
from pathlib import Path, PurePath
from pydoc import safeimport
from typing import Union

import numpy as np
import torch
import random


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


def locate_and_import_class(class_name, models_location: Union[str, PurePath] = 'models', forceload=False):
    """Locate an object by name or dotted path, importing as necessary."""
    models_location = Path(models_location)
    module_paths = [x for x in models_location.rglob('*.py') if x.is_file() and '__init__' not in x.name]
    for module_path in module_paths:
        mod = importlib.import_module('.'.join([x.replace('.py', '') for x in module_path.parts]))
        try:
            model_class = mod.__getattribute__(class_name)
        except AttributeError:
            continue
        return model_class
