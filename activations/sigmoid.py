from activations.base import BaseActivation
import numpy as np

class Sigmoid(BaseActivation):
    def __init__(self):
        super().__init__()

    def forward(self, z, train=None):
        out = 1 / (1 + np.exp(-z))
        if train:
            self.out = out
        return out
    
    def backward(self, error_tensor):
        inner_grad = self.out * (1 - self.out)
        grad = error_tensor * inner_grad
        return grad