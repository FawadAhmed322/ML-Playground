from activations.base import BaseActivation as _BaseActivation
import numpy as np

class Sigmoid(_BaseActivation):
    def __init__(self, name=None):
        super().__init__(name=name)

    def forward(self, z, train=True):
        out = 1 / (1 + np.exp(-z))
        if train:
            self.out = out
        return out
    
    def backward(self, error_tensor):
        inner_grad = self.out * (1 - self.out)
        grad = error_tensor * inner_grad
        return grad