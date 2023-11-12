from optimizers.base import BaseOptimizer as _BaseOptimizer
import numpy as np

class RMSprop(_BaseOptimizer):
    def __init__(self, learning_rate, decay_rate=0.9):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.cache = 0

    def step(self, w, grad):
        self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * np.square(grad)
        w = w - self.learning_rate * grad / (np.sqrt(self.cache) + 1e-8)
        return w
