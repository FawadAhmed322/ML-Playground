import numpy as np

def mean_squared_error(y_pred, y_true, train=False):
    loss = np.sum(np.square(y_pred - y_true), axis=0)
    if not train:
        return loss
    else:
        d_loss = 2 * (y_pred - y_true)
        return loss, d_loss