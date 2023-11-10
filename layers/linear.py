import numpy as np
from layers.base import BaseLayer

class Linear(BaseLayer):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
    
    def forward(self, x):
        if self.bias:
            x = np.hstack([np.ones((x.shape[0], 1)), x])
        self.x = x
        out = np.dot(x, self.w)
        return out

    def backward(self, error_tensor):
        grad = np.dot(self.x.T, error_tensor)
        return grad
    
if __name__ == '__main__':
    linear_layer = Linear(3, 3)
    print(linear_layer.w.shape)