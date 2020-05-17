from torchvision.transforms import ToTensor as TorchVisionToTensor


class ToTensor(TorchVisionToTensor):

    def __call__(self, pic):
        tensor = super(ToTensor, self).__call__(pic).float()
        return tensor
