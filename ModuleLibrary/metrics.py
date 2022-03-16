#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 1 10:01 2022
@author: lou

	This module contains the custom losses or metrics that can be used to train or to evaluate a neural network.
	It is made to work as a usual loss or metric.
"""

import tensorflow.keras.backend as K
from tensorflow.python.ops import math_ops
from sklearn.metrics import  confusion_matrix
import math



def MCC(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def recall(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))

    tp = K.sum(y_pos * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    return tp / (tp + fn)

def precision(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos) + 1   
    fp = K.sum(y_neg * y_pred_pos) + 1

    return tp / (tp + fp)

def BA(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    p = tp + fn
    n = fp + tn
    
    tpr = math_ops.div_no_nan(tp, p)
    tnr = math_ops.div_no_nan(tn, n)

    return (tpr + tnr) / 2






