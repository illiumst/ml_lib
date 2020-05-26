import pickle
import shelve
from pathlib import Path


def fix_all_random_seeds(config_obj):
    import numpy as np
    import torch
    import random
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