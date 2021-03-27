import numpy as np

import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score, roc_auc_score, average_precision_score

from ml_lib.metrics._base_score import _BaseScores
from ml_lib.utils.tools import to_one_hot


class BinaryScores(_BaseScores):

    def __init__(self, *args):
        super(BinaryScores, self).__init__(*args)

    def __call__(self, outputs):
        summary_dict = dict()

        # Additional Score like the unweighted Average Recall:
        #########################
        # INIT
        if isinstance(outputs['batch_y'], torch.Tensor):
            y_true = outputs['batch_y'].cpu().numpy()
        else:
            y_true = torch.cat([output['batch_y'] for output in outputs]).cpu().numpy()

        if isinstance(outputs['y'], torch.Tensor):
            y_pred = outputs['y'].cpu().numpy()
        else:
            y_pred = torch.cat([output['y'] for output in outputs]).squeeze().cpu().float().numpy()

        # UnweightedAverageRecall
        # y_true = torch.cat([output['batch_y'] for output in outputs]).cpu().numpy()
        # y_pred = torch.cat([output['element_wise_recon_error'] for output in outputs]).squeeze().cpu().numpy()

        # How to apply a threshold manualy
        # y_pred = (y_pred >= 0.5).astype(np.float32)

        # How to apply a threshold by IF (Isolation Forest)
        clf = IsolationForest()
        y_score = clf.fit_predict(y_pred.reshape(-1, 1))
        y_score = (np.asarray(y_score) == -1).astype(np.float32)

        uar_score = recall_score(y_true, y_score, labels=[0, 1], average='macro',
                                 sample_weight=None, zero_division='warn')
        summary_dict.update(dict(uar_score=uar_score))
        #########################
        # Precission
        precision_score = average_precision_score(y_true, y_score)
        summary_dict.update(dict(precision_score=precision_score))

        #########################
        # AUC
        try:
            auc_score = roc_auc_score(y_true=y_true, y_score=y_score)
            summary_dict.update(dict(auc_score=auc_score))
        except ValueError:
            summary_dict.update(dict(auc_score=-1))

        #########################
        # pAUC
        try:
            pauc = roc_auc_score(y_true=y_true, y_score=y_score, max_fpr=0.15)
            summary_dict.update(dict(pauc_score=pauc))
        except ValueError:
            summary_dict.update(dict(pauc_score=-1))

        return summary_dict