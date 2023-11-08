import numpy as np
def MSELoss(y_preds, y_true, train=False):
    epsilon = 1e-15  # Small value to prevent log(0)
    y_preds = np.clip(y_preds, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
    square = np.square(y_preds - y_true)
    loss = np.sum(square, axis=0)
    d_loss = None
    if train:
        d_loss = 2 * (y_preds - y_true)
    return loss, d_loss