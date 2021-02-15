import numpy as np

from einops import reduce


import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import recall_score, roc_auc_score, average_precision_score

from ml_lib.metrics._base_score import _BaseScores


class AttentionRollout(_BaseScores):

    def __init__(self, *args):
        super(AttentionRollout, self).__init__(*args)
        pass

    def __call__(self, outputs):
        summary_dict = dict()
        #######################################################################################
        # Additional Score  -  Histogram Distances - Image Plotting
        #######################################################################################
        #
        # INIT
        attn_weights = [output['attn_weights'].cpu().numpy() for output in outputs]
        attn_reduce_heads = [reduce(x, '') for x in attn_weights]

        if self.model.params.use_residual:
            residual_att = np.eye(att_mat.shape[1])[None, ...]
            aug_att_mat = att_mat + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(axis=-1)[..., None]
        else:
            aug_att_mat = att_mat

        joint_attentions = np.zeros(aug_att_mat.shape)

        layers = joint_attentions.shape[0]
        joint_attentions[0] = aug_att_mat[0]
        for i in np.arange(1, layers):
            joint_attentions[i] = aug_att_mat[i].dot(joint_attentions[i - 1])






