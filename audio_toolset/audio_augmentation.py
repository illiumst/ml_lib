try:
    import librosa
except ImportError:  # pragma: no-cover
    raise ImportError('You want to use `librosa` plugins which are not installed yet,'  # pragma: no-cover
                      ' install it with `pip install librosa`.')

import numpy as np


class Speed(object):

    def __init__(self, max_amount=0.3, speed_min=1, speed_max=1):
        self.speed_max = speed_max if speed_max else 1
        self.speed_min = speed_min if speed_min else 1
        # noinspection PyTypeChecker
        self.max_amount = min(max(0, max_amount), 1)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__})'

    def __call__(self, x):
        if self.speed_min == 1 and self.speed_max == 1:
            return x
        start = int(np.random.randint(low=0, high=x.shape[-1], size=1))
        width = np.random.uniform(low=0, high=self.max_amount, size=1) * x.shape[-1]
        end = int(width + start)
        end = min(end, x.shape[-1])
        try:
            speed_factor = float(np.random.uniform(low=self.speed_min, high=self.speed_max, size=1))
            aug_data = librosa.effects.time_stretch(y=x[start:end], rate=speed_factor)
            x_aug = np.concatenate((x[:start], aug_data, x[end:]), axis=0)[:x.shape[-1]]
            if speed_factor > 1:
                embedding = np.zeros_like(x)
                embedding[:x_aug.shape[0]] = x_aug
                x_aug = embedding
            return x_aug
        except ValueError:
            return x
