import numpy as np
from sklearn.metrics import roc_auc_score


def calc_auc(pred, label):
    return roc_auc_score(label, pred)