from abc import ABC


class _BaseScores(ABC):

    def __init__(self, lightning_model):
        self.model = lightning_model
        pass

    def __call__(self, outputs):
        # summary_dict = dict()
        # return summary_dict
        raise NotImplementedError