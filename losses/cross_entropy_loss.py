import numpy as np

class CrossEntropyLoss:
    def __init__(self, binary=False, n_classes=None):
        self.epsilon = 1e-15  # Small value to prevent log(0)
        self.binary = binary
        self.n_classes = n_classes

    def forward(self, y_preds, y_true, train=True):
        if self.n_classes is None:
            self.n_classes = y_true.shape[1]
        y_preds = np.clip(y_preds, self.epsilon, 1 - self.epsilon)  # Clip values to avoid log(0)
        if train:
            self.y_preds = y_preds
            self.y_true = y_true
        if self.binary:
            loss = -np.sum(y_true * np.log(y_preds) + (1 - y_true) * np.log(1 - y_preds))
        else:
            if train:
                self.y_true = y_true
            loss = -np.sum(y_true * np.log(y_preds))
        return loss
    
    def backward(self):
        if self.binary:
            grad = (self.y_preds - self.y_true) / (self.y_preds * (1 - self.y_preds))
        else:
            grad = self.y_preds - self.y_true
        return grad

if __name__ == '__main__':
    loss_fn = CrossEntropyLoss()
    y_true = np.array([[0, 1, 0]])
    y_preds = np.array([[0.1, 0.6, 0.3]])
    loss = loss_fn.forward(y_preds, y_true)
    grad = loss_fn.backward()
    print(loss, grad)