from optimizers.base import BaseOptimizer as _BaseOptimizer
import numpy as np

class Adagrad(_BaseOptimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.cache = 0

    def step(self, w, grad):
        self.cache += np.square(grad)
        w = w - self.learning_rate * grad / (np.sqrt(self.cache) + 1e-8)
        return w
