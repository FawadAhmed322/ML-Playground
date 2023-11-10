import numpy as np
class MSELoss:
    def __init__(self):
        self.epsilon = 1e-15  # Small value to prevent log(0)

    def forward(self, y_preds, y_true):
        y_preds = np.clip(y_preds, self.epsilon, 1 - self.epsilon)  # Clip values to avoid log(0)
        self.diff = y_preds - y_true
        square = np.square(self.diff)
        loss = np.mean(square, axis=0)
        return loss

    def backward(self):
        grad = 2 * self.diff
        return grad