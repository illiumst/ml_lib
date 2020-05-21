try:
    import librosa
except ImportError:  # pragma: no-cover
    raise ImportError('You want to use `librosa` plugins which are not installed yet,'  # pragma: no-cover
                      ' install it with `pip install librosa`.')

import numpy as np


class Speed(object):

    def __init__(self, max_amount=0.3, speed_min=1, speed_max=1):
        self.speed_max = speed_max
        self.speed_min = speed_min
        self.max_amount = max_amount

    def __call__(self, x):
        assert all([self.speed_min, self.speed_max, self.max_amount])
        start = int(np.random.randint(low=0, high=x.shape[-1], size=1))
        width = np.random.uniform(low=0, high=self.max_amount, size=1) * x.shape[-1]
        end = int(width + start)
        end = min(end, x.shape[-1])
        try:
            speed_factor = float(np.random.uniform(low=self.speed_min, high=self.speed_max, size=1))
            aug_data = librosa.effects.time_stretch(x[start:end], speed_factor)
            return np.concatenate((x[:start], aug_data, x[end:]), axis=0)[:x.shape[-1]]
        except ValueError:
            return x
