from abc import ABC
from torchvision.transforms import ToTensor as TorchVisionToTensor


class _BaseTransformation(ABC):

    def __init__(self, *args):
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__})'

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class ToTensor(TorchVisionToTensor):

    def __call__(self, pic):
        # Make it float .float() == 32bit
        tensor = super(ToTensor, self).__call__(pic).float()
        return tensor
