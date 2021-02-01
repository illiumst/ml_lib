from itertools import cycle

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix
from scipy.spatial.distance import cdist

from ml_lib.metrics._base_score import _BaseScores

from matplotlib import pyplot as plt


class GenerativeTaskEval(_BaseScores):

    def __init__(self, *args):
        super(GenerativeTaskEval, self).__init__(*args)
        pass

    def __call__(self, outputs):
        summary_dict = dict()
        #######################################################################################
        # Additional Score  -  Histogram Distances - Image Plotting
        #######################################################################################
        #
        # INIT
        y_true = torch.cat([output['batch_y'] for output in outputs]).cpu().numpy()

        y_pred = torch.cat([output['y'] for output in outputs]).squeeze().cpu().numpy()

        attn_weights = torch.cat([output['attn_weights'] for output in outputs]).squeeze().cpu().numpy()

        ######################################################################################
        #
        # Histogram comparission

        y_true_hist = np.histogram(y_true, bins=128)[0]  # Todo: Find a better value
        y_pred_hist = np.histogram(y_pred, bins=128)[0]  # Todo: Find a better value

        # L2 norm == euclidean distance
        hist_euc_dist = cdist(np.expand_dims(y_true_hist, axis=0), np.expand_dims(y_pred_hist, axis=0),
                              metric='euclidean')

        # Manhattan Distance
        hist_manhattan_dist = cdist(np.expand_dims(y_true_hist, axis=0), np.expand_dims(y_pred_hist, axis=0),
                                    metric='cityblock')

        summary_dict.update(hist_manhattan_dist=hist_manhattan_dist, hist_euc_dist=hist_euc_dist)

        #######################################################################################
        #
        idx = np.random.choice(np.arange(y_true.shape[0]), 1).item()

        ax = plt.imshow(y_true[idx].squeeze())
        # Plot using a small number of colors, with unevenly spaced boundaries.
        ax2 = plt.imshow(attn_weights[idx].sq, interpolation='nearest', aspect='auto', extent=ax.get_extent())
        self.model.logger.log_image('ROC', image=plt.gcf(), step=self.model.current_epoch)
        plt.clf()


        #######################################################################################
        #


        #######################################################################################
        #

        plt.close('all')
        return summary_dict