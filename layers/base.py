import numpy as np

class BaseLayer:
    def __init__(self, in_features, out_features, bias=True):
        self.bias = bias
        if bias:
            in_features = in_features + 1
        self.w = np.random.rand(in_features, out_features)
        
    def forward(self, x):
        pass
    
    def backward(self, error_tensor):
        pass