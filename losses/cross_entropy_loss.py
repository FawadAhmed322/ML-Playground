import numpy as np
from utils import indices_to_one_hot
def CrossEntropyLoss(y_preds, y_true, n_classes, train=False):
    epsilon = 1e-15  # Small value to prevent log(0)
    y_preds = np.clip(y_preds, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
    y_one_hot = indices_to_one_hot(y_true, n_classes)
    loss = -np.sum(y_one_hot * np.log(y_preds))
    d_loss = None
    if train:
        d_loss = y_one_hot / y_preds
    return loss, d_loss