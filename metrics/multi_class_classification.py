from itertools import cycle

import numpy as np
import torch
from pytorch_lightning.metrics import Recall
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score, ConfusionMatrixDisplay, confusion_matrix, \
    recall_score

from ml_lib.metrics._base_score import _BaseScores
from ml_lib.utils.tools import to_one_hot

from matplotlib import pyplot as plt


class MultiClassScores(_BaseScores):

    def __init__(self, *args):
        super(MultiClassScores, self).__init__(*args)
        pass

    def __call__(self, outputs, class_names=None):
        summary_dict = dict()
        class_names = class_names or range(self.model.params.n_classes)
        #######################################################################################
        # Additional Score  -  UAR  -  ROC  -  Conf. Matrix  -  F1
        #######################################################################################
        #
        # INIT
        if isinstance(outputs['batch_y'], torch.Tensor):
            y_true = outputs['batch_y'].cpu().numpy()
        else:
            y_true = torch.cat([output['batch_y'] for output in outputs]).cpu().numpy()
        y_true_one_hot = to_one_hot(y_true, self.model.params.n_classes)

        if isinstance(outputs['y'], torch.Tensor):
            y_pred = outputs['y'].cpu().numpy()
        else:
            y_pred = torch.cat([output['y'] for output in outputs]).squeeze().cpu().float().numpy()
        y_pred_max = np.argmax(y_pred, axis=1)

        class_names = {val: key for val, key in enumerate(class_names)}
        ######################################################################################
        #
        # F1 SCORE
        micro_f1_score = f1_score(y_true, y_pred_max, labels=None, pos_label=1, average='micro', sample_weight=None,
                                  zero_division=True)
        macro_f1_score = f1_score(y_true, y_pred_max, labels=None, pos_label=1, average='macro', sample_weight=None,
                                  zero_division=True)
        summary_dict.update(dict(micro_f1_score=micro_f1_score, macro_f1_score=macro_f1_score))
        ######################################################################################
        #
        # Unweichted Average Recall
        uar = recall_score(y_true, y_pred_max, labels=[0, 1, 2, 3, 4], average='macro',
                           sample_weight=None, zero_division='warn')
        summary_dict.update(dict(uar_score=uar))
        #######################################################################################
        #
        # ROC Curve

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.model.params.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_one_hot.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.model.params.n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.model.params.n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= self.model.params.n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label=f'micro ROC ({round(roc_auc["micro"], 2)})',
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label=f'macro ROC({round(roc_auc["macro"], 2)})',
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['firebrick', 'orangered', 'gold', 'olive', 'limegreen', 'aqua',
                        'dodgerblue', 'slategrey', 'royalblue', 'indigo', 'fuchsia'], )

        for i, color in zip(range(self.model.params.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'{class_names[i]} ({round(roc_auc[i], 2)})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")

        self.model.logger.log_image('ROC', image=plt.gcf(), step=self.model.current_epoch)
        # self.model.logger.log_image('ROC', image=plt.gcf(), step=self.model.current_epoch, ext='pdf')
        plt.clf()

        #######################################################################################
        #
        # ROC AUC SCORE

        try:
            macro_roc_auc_ovr = roc_auc_score(y_true_one_hot, y_pred, multi_class="ovr",
                                              average="macro")
            summary_dict.update(macro_roc_auc_ovr=macro_roc_auc_ovr)
        except ValueError:
            micro_roc_auc_ovr = roc_auc_score(y_true_one_hot, y_pred, multi_class="ovr",
                                              average="micro")
            summary_dict.update(micro_roc_auc_ovr=micro_roc_auc_ovr)

        #######################################################################################
        #
        # Confusion matrix
        fig1, ax1 = plt.subplots(dpi=96)
        cm = confusion_matrix([class_names[x] for x in y_true], [class_names[x] for x in y_pred_max],
                              labels=[class_names[key] for key in class_names.keys()],
                              normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=[class_names[i] for i in range(self.model.params.n_classes)]
                                      )
        disp.plot(include_values=True, ax=ax1)

        self.model.logger.log_image('Confusion_Matrix', image=fig1, step=self.model.current_epoch)
        # self.model.logger.log_image('Confusion_Matrix', image=disp.figure_, step=self.model.current_epoch, ext='pdf')

        plt.close('all')
        return summary_dict
