import numpy as np

class BaseLayer:
    def __init__(self, in_features, out_features, bias=True, trainable=True, optimizer=None, name=None):
        self._name = name
        self._trainable = trainable
        self._optimizer = optimizer
        self.bias = bias
        if bias:
            in_features = in_features + 1
        self.w = np.random.rand(in_features, out_features)
        
    def forward(self, x):
        pass
    
    def backward(self, error_tensor):
        pass

    def is_trainable(self):
        return self._trainable
    
    def freeze(self):
        self._trainable = False

    def unfreeze(self):
        self._trainable = True

    def get_name(self):
        return self._name
    
    def set_name(self, name):
        self._name = name

    def add_optimizer(self, optimizer):
        self._optimizer = optimizer