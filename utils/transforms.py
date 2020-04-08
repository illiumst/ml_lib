import numpy as np


class AsArray(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, x):
        array = np.zeros((self.width, self.height))
        return array
