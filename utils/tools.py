import pickle
import shelve
from pathlib import Path

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
