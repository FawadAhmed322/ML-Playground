from activations.base import BaseActivation as _BaseActivation

class ReLU(_BaseActivation):
    def __init__(self, name=None):
        super().__init__(name=name)

    def forward(self, z, train=True):
        z[z < 0] = 0
        out = z
        if train:
            self.out = out
        return out
    
    def backward(self, error_tensor):
        grad = error_tensor * (self.out > 0)
        return grad