from typing import Union

import numpy as np


class Normalize(object):

    def __init__(self, min_db_level: Union[int, float]):
        self.min_db_level = min_db_level

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return np.clip((s - self.min_db_level) / -self.min_db_level, 0, 1)


class DeNormalize(object):

    def __init__(self, min_db_level: Union[int, float]):
        self.min_db_level = min_db_level

    def __call__(self, s: np.ndarray) -> np.ndarray:
        return (np.clip(s, 0, 1) * -self.min_db_level) + self.min_db_level
