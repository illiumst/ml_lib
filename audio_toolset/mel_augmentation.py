import numpy as np

from ml_lib.utils.transforms import _BaseTransformation


class NoiseInjection(_BaseTransformation):

    def __init__(self, noise_factor: float, sigma=1, mu=0):
        super(NoiseInjection, self).__init__()
        assert noise_factor >= 0, f'noise_factor has to be greater then 0, but was: {noise_factor}.'
        self.mu = mu
        self.sigma = sigma
        self.noise_factor = noise_factor

    def __call__(self, x: np.ndarray):
        if self.noise_factor:
            noise = np.random.normal(self.mu, self.sigma, size=x.shape) * self.noise_factor
            augmented_data = x + x * noise
            # Cast back to same data type
            augmented_data = augmented_data.astype(x.dtype)
            return augmented_data
        else:
            return x


class LoudnessManipulator(_BaseTransformation):

    def __init__(self, max_factor: float):
        super(LoudnessManipulator, self).__init__()
        assert 1 > max_factor >= 0, f'max_shift_ratio has to be between [0,1], but was: {max_factor}.'

        self.max_factor = max_factor

    def __call__(self, x):
        if self.max_factor:
            augmented_data = x + x * (np.random.random() * self.max_factor)
            # Cast back to same data type
            augmented_data = augmented_data.astype(x.dtype)
            return augmented_data
        else:
            return x


class ShiftTime(_BaseTransformation):

    valid_shifts = ['right', 'left', 'any']

    def __init__(self, max_shift_ratio: float, shift_direction: str = 'any'):
        super(ShiftTime, self).__init__()
        assert 1 > max_shift_ratio >= 0, f'max_shift_ratio has to be between [0,1], but was: {max_shift_ratio}.'
        assert shift_direction.lower() in self.valid_shifts, f'shift_direction has to be one of: {self.valid_shifts}'
        self.max_shift_ratio = max_shift_ratio
        self.shift_direction = shift_direction.lower()

    def __call__(self, x: np.ndarray):
        if self.max_shift_ratio:
            shift = np.random.randint(max(int(self.max_shift_ratio * x.shape[-1]), 1))
            if self.shift_direction == 'right':
                shift = -1 * shift
            elif self.shift_direction == 'any':
                direction = np.asscalar(np.random.choice([1, -1], 1))
                shift = direction * shift
            augmented_data = np.roll(x, shift)
            # Set to silence for heading/ tailing
            shift = int(shift)
            if shift > 0:
                augmented_data[:shift, :] = 0
            else:
                augmented_data[shift:, :] = 0
            return augmented_data
        else:
            return x


class MaskAug(_BaseTransformation):

    w_idx = -1
    h_idx = -2

    def __init__(self, duration_ratio_max=0.3, mask_with_noise=True):
        super(MaskAug, self).__init__()
        assertion = f'"duration_ratio" has to be within [0..1], but was: {duration_ratio_max}'
        if isinstance(duration_ratio_max, (tuple, list)):
            assert all([0 < max_val < 1 for max_val in duration_ratio_max]), assertion
        if isinstance(duration_ratio_max, (float, int)):
            assert 0 <= duration_ratio_max < 1, assertion
        super().__init__()

        self.mask_with_noise = mask_with_noise
        self.duration_ratio_max = duration_ratio_max if isinstance(duration_ratio_max, (tuple, list)) \
            else (duration_ratio_max, duration_ratio_max)

    def __call__(self, x):
        for dim in (self.w_idx, self.h_idx):
            if self.duration_ratio_max[dim]:
                start = np.asscalar(np.random.choice(x.shape[dim], 1))
                v_max = int(x.shape[dim] * self.duration_ratio_max[dim])
                size = np.asscalar(np.random.randint(0, v_max, 1))
                end = int(min(start + size, x.shape[dim]))
                size = end - start
                if dim == self.w_idx:
                    x[:, start:end] = np.random.random((x.shape[self.h_idx], size)) if self.mask_with_noise else 0
                else:
                    x[start:end, :] = np.random.random((size, x.shape[self.w_idx])) if self.mask_with_noise else 0
        return x
