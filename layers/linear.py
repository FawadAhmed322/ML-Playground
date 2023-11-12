from layers.base import BaseLayer as _BaseLayer
import numpy as np

class Linear(_BaseLayer):
    def __init__(self, in_features, out_features, bias=True, trainable=True, optimizer=None, name=None):
        super().__init__(in_features, out_features, bias, trainable, optimizer, name)
    
    def forward(self, x, train=True):
        if self.bias:
            x = np.hstack([np.ones((x.shape[0], 1)), x])
        if train:
            self._x = x
        out = np.dot(x, self.w)
        return out

    def backward(self, error_tensor):
        grad_input = np.dot(error_tensor, self.w.T)[:, 1:]
        if self.is_trainable():
            if self._optimizer is not None:
                grad_weights = np.dot(self._x.T, error_tensor)
                self.w = self._optimizer.step(self.w, grad_weights)
            else:
                raise Exception("Trainable Layer must have optimizer passed to it.")
        return grad_input
    
if __name__ == '__main__':
    linear_layer = Linear(3, 3)
    print(linear_layer.w.shape)