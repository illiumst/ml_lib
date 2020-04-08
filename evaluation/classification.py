import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


class ROCEvaluation(object):

    linewidth = 2

    def __init__(self, plot_roc=False):
        self.plot_roc = plot_roc
        self.epoch = 0

    def __call__(self, prediction, label):

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(prediction, label)
        roc_auc = auc(fpr, tpr)
        if self.plot_roc:
            _ = plt.gcf()
            plt.plot(fpr, tpr, color='darkorange', lw=self.linewidth, label=f'ROC curve (area = {roc_auc})')
            self._prepare_fig()
        return roc_auc, fpr, tpr

    def _prepare_fig(self):
        fig = plt.gcf()
        ax = plt.gca()
        plt.plot([0, 1], [0, 1], color='navy', lw=self.linewidth, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        fig.legend(loc="lower right")

        return fig
