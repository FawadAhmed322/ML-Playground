class BaseActivation:
    def __init__(self):
        self._trainable = False
        
    def forward(self, x):
        pass
    
    def backward(self, error_tensor):
        pass

    def is_trainable(self):
        return self._trainable