import librosa
import torch
from scipy.signal import butter, lfilter

from ml_lib.modules.utils import AutoPad
import numpy as np

def butter_lowpass(cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    # noinspection PyTupleAssignmentBalance
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a


def butter_lowpass_filter(data, cutoff, sr, order=5):
    b, a = butter_lowpass(cutoff, sr, order=order)
    y = lfilter(b, a, data)
    return y


class MFCC(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, y):
        mfcc = librosa.feature.mfcc(y, **self.__dict__)
        return mfcc


class NormalizeLocal(object):
    def __init__(self):
        self.cache: torch.Tensor
        pass

    def __call__(self, x: torch.Tensor):
        mean = x.mean()
        std = x.std()

        x = x.__sub__(mean).__div__(std)
        x[torch.isnan(x)] = 0
        x[torch.isinf(x)] = 0
        return x


class NormalizeMelband(object):
    def __init__(self):
        self.cache: torch.Tensor
        pass

    def __call__(self, x: torch.Tensor):
        mean = x.mean(-1).unsqueeze(-1)
        std = x.std(-1).unsqueeze(-1)

        x = x.__sub__(mean).__div__(std)
        x[torch.isnan(x)] = 0
        x[torch.isinf(x)] = 0
        return x


class AutoPadToShape(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, x):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        embedding = torch.zeros(self.shape)
        embedding[: x.shape] = x
        return embedding

    def __repr__(self):
        return f'AutoPadTransform({self.shape})'


class Melspectogram(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, y):
        mel = librosa.feature.melspectrogram(y, **self.__dict__)
        mel = librosa.amplitude_to_db(mel, ref=np.max)
        return mel

    def __repr__(self):
        return f'MelSpectogram({self.__dict__})'


class LowPass(object):
    def __init__(self, sr=16000):
        self.sr = sr

    def __call__(self, x):
        return butter_lowpass_filter(x, 1000, 1)