try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no-cover
    raise ImportError('You want to use `matplotlib` plugins which are not installed yet,'  # pragma: no-cover
                      ' install it with `pip install matplotlib`.')
try:
    from sklearn.metrics import roc_curve, auc, recall_score
except ImportError:  # pragma: no-cover
    raise ImportError('You want to use `sklearn` plugins which are not installed yet,'  # pragma: no-cover
                      ' install it with `pip install scikit-learn`.')


class ROCEvaluation(object):

    linewidth = 2

    def __init__(self, plot=False):
        self.plot = plot
        self.epoch = 0

    def __call__(self, prediction, label):

        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(prediction, label)
        roc_auc = auc(fpr, tpr)
        if self.plot:
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


class UAREvaluation(object):

    def __init__(self, labels: list, plot=False):
        self.labels = labels
        self.plot_roc = plot
        self.epoch = 0

    def __call__(self, prediction, label):

        # Compute uar score - UnweightedAverageRecal

        uar_score = recall_score(label, prediction, labels=self.labels, average='macro',
                                 sample_weight=None, zero_division='warn')
        return uar_score

    def _prepare_fig(self):
        raise NotImplementedError # TODO Implement a nice visualization
        fig = plt.gcf()
        ax = plt.gca()
        plt.plot([0, 1], [0, 1], color='navy', lw=self.linewidth, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        fig.legend(loc="lower right")

        return fig