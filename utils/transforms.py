from torchvision.transforms import ToTensor as TorchvisionToTensor


class ToTensor(TorchvisionToTensor):

    def __call__(self, pic):
        tensor = super(ToTensor, self).__call__(pic).float()
        return tensor