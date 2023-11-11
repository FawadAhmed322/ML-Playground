import numpy as np
from utils.utils import indices_to_one_hot

class CrossEntropyLoss:
    def __init__(self, binary=False, n_classes=None):
        self.epsilon = 1e-15  # Small value to prevent log(0)
        self.binary = binary
        self.n_classes = n_classes
        if not binary:
            if n_classes is None or n_classes < 2:
                raise Exception("if binary flag is False and must be n_classes must be >= 2")

    def forward(self, y_preds, y_true, train=True):
        y_preds = np.clip(y_preds, self.epsilon, 1 - self.epsilon)  # Clip values to avoid log(0)
        if train:
            self.y_preds = y_preds
            self.y_true = y_true
        if self.binary:
            loss = -np.sum(y_true * np.log(y_preds) + (1 - y_true) * np.log(1 - y_preds))
        else:
            y_one_hot = indices_to_one_hot(y_true, self.n_classes)
            if train:
                self.y_one_hot = y_one_hot
            loss = -np.sum(y_one_hot * np.log(y_preds))
        return loss
    
    def backward(self):
        if self.binary:
            grad = (self.y_preds - self.y_true) / (self.y_preds * (1 - self.y_preds))
        else:
            grad = self.y_one_hot / self.y_preds
        return grad

if __name__ == '__main__':
    loss_fn = CrossEntropyLoss(binary=True)
    y_true = np.array([1])
    y_preds = np.array([0.65])
    loss = loss_fn.forward(y_preds, y_true)
    grad = loss_fn.backward()
    print(loss, grad)