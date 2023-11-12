class BaseActivation:
    def __init__(self, name=None):
        self._trainable = False
        self._name = name
        
    def forward(self, x):
        pass
    
    def backward(self, error_tensor):
        pass

    def is_trainable(self):
        return self._trainable
    
    def get_name(self):
        return self._name
    
    def set_name(self, name):
        self._name = name