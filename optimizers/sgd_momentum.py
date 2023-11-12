from optimizers.base import BaseOptimizer as _BaseOptimizer

class SGDMomentum(_BaseOptimizer):
    """Stochastic Gradient Descent with Momentum"""
    def __init__(self, learning_rate, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v = 0

    def step(self, w, grad):
        self.v = self.momentum * self.v - self.learning_rate * grad
        w = w + self.v
        return w