import librosa
from scipy.signal import butter, lfilter

import numpy as np


def scale_minmax(x, min_val=0.0, max_val=1.0):
    x_std = (x - x.min()) / (x.max() - x.min())
    x_scaled = x_std * (max_val - min_val) + min_val
    return x_scaled


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
        self.cache: np.ndarray
        pass

    def __call__(self, x: np.ndarray):
        mean = x.mean()
        std = x.std() + 0.0001

        # Pytorch Version:
        # x = x.__sub__(mean).__div__(std)
        # Numpy Version
        x = (x - mean) / std
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        return x


class NormalizeMelband(object):
    def __init__(self):
        self.cache: np.ndarray
        pass

    def __call__(self, x: np.ndarray):
        mean = x.mean(-1).unsqueeze(-1)
        std = x.std(-1).unsqueeze(-1)

        x = x.__sub__(mean).__div__(std)
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        return x


class AudioToMel(object):
    def __init__(self, amplitude_to_db=False, power_to_db=False, **kwargs):
        assert not all([amplitude_to_db, power_to_db]), "Choose amplitude_to_db or power_to_db, not both!"
        self.mel_kwargs = kwargs
        self.amplitude_to_db = amplitude_to_db
        self.power_to_db = power_to_db

    def __call__(self, y):
        mel = librosa.feature.melspectrogram(y, **self.mel_kwargs)
        if self.amplitude_to_db:
            mel = librosa.amplitude_to_db(mel, ref=np.max)
        if self.power_to_db:
            mel = librosa.power_to_db(mel, ref=np.max)
        return mel

    def __repr__(self):
        return f'MelSpectogram({self.__dict__})'


class PowerToDB(object):
    def __init__(self, running_max=False):
        self.running_max = 0 if running_max else None

    def __call__(self, x):
        if self.running_max is not None:
            self.running_max = max(np.max(x), self.running_max)
            return librosa.power_to_db(x, ref=self.running_max)
        return librosa.power_to_db(x, ref=np.max)


class LowPass(object):
    def __init__(self, sr=16000):
        self.sr = sr

    def __call__(self, x):
        return butter_lowpass_filter(x, 1000, 1)


class MelToImage(object):
    def __init__(self):
        pass

    def __call__(self, x):
        # Source to Solution: https://stackoverflow.com/a/57204349
        mels = np.log(x + 1e-9)  # add small number to avoid log(0)

        # min-max scale to fit inside 8-bit range
        img = scale_minmax(mels, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0)  # put low frequencies at the bottom in image
        img = 255 - img  # invert. make black==more energy
        return img
