from torchvision.transforms import ToTensor as TorchVisionToTensor


class ToTensor(TorchVisionToTensor):

    def __call__(self, pic):
        # Make it float .float() == 32bit
        tensor = super(ToTensor, self).__call__(pic).float()
        return tensor
