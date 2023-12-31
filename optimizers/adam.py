from optimizers.base import BaseOptimizer as _BaseOptimizer
import numpy as np

class Adam(_BaseOptimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = 0
        self.v = 0
        self.t = 0

    def step(self, w, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grad)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        w = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        return w
