import numpy as np
def MSELoss(y_pred, y_true, train=False):
    square = np.square(y_pred - y_true)
    loss = np.sum(square, axis=0)
    d_loss = None
    if train:
        d_loss = 2 * (y_pred - y_true)
    return loss, d_loss