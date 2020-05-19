try:
    import librosa
except ImportError:  # pragma: no-cover
    raise ImportError('You want to use `librosa` plugins which are not installed yet,'  # pragma: no-cover
                      ' install it with `pip install librosa`.')

import numpy as np


class Speed(object):

    def __init__(self, max_ratio=0.3, speed_factor=1):
        self.speed_factor = speed_factor
        self.max_ratio = max_ratio

    def __call__(self, x):
        if not all([self.speed_factor, self.max_ratio]):
            return x
        start = int(np.random.randint(0, x.shape[-1],1))
        end = int((np.random.uniform(0, self.max_ratio, 1) * x.shape[-1]) + start)
        end = min(end, x.shape[-1])
        try:
            speed_factor = float(np.random.uniform(min(self.speed_factor, 1), max(self.speed_factor, 1), 1))
            aug_data = librosa.effects.time_stretch(x[start:end], speed_factor)
            return np.concatenate((x[:start], aug_data, x[end:]), axis=0)[:x.shape[-1]]
        except ValueError:
            return x
