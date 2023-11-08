import numpy as np
from utils import indices_to_one_hot
def CrossEntropyLoss(y_preds, y_true, n_classes, binary=False, train=False):
    epsilon = 1e-15  # Small value to prevent log(0)
    y_preds = np.clip(y_preds, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
    loss = None
    d_loss = None
    if not binary:
        y_one_hot = indices_to_one_hot(y_true, n_classes)
        loss = -np.sum(y_one_hot * np.log(y_preds))
        if train:
            d_loss = y_one_hot / y_preds
    else:
        loss = -np.sum(y_true * np.log(y_preds) + (1 - y_true) * np.log(1 - y_preds))
        d_loss = y_true / y_preds - (1 - y_true) / (1 - y_preds)
    return loss, d_loss