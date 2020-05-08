import numpy as np


class NoiseInjection(object):

    def __init__(self, noise_factor: float, sigma=0.5, mu=0.5):
        assert noise_factor > 0, f'max_shift_ratio has to be greater then 0, but was: {noise_factor}.'
        self.mu = mu
        self.sigma = sigma
        self.noise_factor = noise_factor

    def __call__(self, x: np.ndarray):
        noise = np.random.normal(loc=self.mu, scale=self.sigma, size=x.shape)
        augmented_data = x + self.noise_factor * noise
        # Cast back to same data type
        augmented_data = augmented_data.astype(x.dtype)
        return augmented_data


class LoudnessManipulator(object):

    def __init__(self, max_factor: float):
        assert 1 > max_factor > 0, f'max_shift_ratio has to be between [0,1], but was: {max_factor}.'

        self.max_factor = max_factor

    def __call__(self, x: np.ndarray):
        augmented_data = x + x * (np.random.random() * self.max_factor)
        # Cast back to same data type
        augmented_data = augmented_data.astype(x.dtype)
        return augmented_data


class ShiftTime(object):

    valid_shifts = ['right', 'left', 'any']

    def __init__(self, max_shift_ratio: float, shift_direction: str = 'any'):
        assert 1 > max_shift_ratio > 0, f'max_shift_ratio has to be between [0,1], but was: {max_shift_ratio}.'
        assert shift_direction.lower() in self.valid_shifts, f'shift_direction has to be one of: {self.valid_shifts}'
        self.max_shift_ratio = max_shift_ratio
        self.shift_direction = shift_direction.lower()

    def __call__(self, x: np.ndarray):
        shift = np.random.randint(max(int(self.max_shift_ratio * x.shape[-1]), 1))
        if self.shift_direction == 'right':
            shift = -1 * shift
        elif self.shift_direction == 'any':
            direction = np.random.choice([1, -1], 1)
            shift = direction * shift
        augmented_data = np.roll(x, shift)
        # Set to silence for heading/ tailing
        shift = int(shift)
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        return augmented_data
