from optimizers.base import BaseOptimizer as _BaseOptimizer

class NAG(_BaseOptimizer):
    """Nesterov Accelerated Gradient"""
    def __init__(self, learning_rate, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v = 0

    def step(self, w, grad):
        self.v = self.momentum * self.v - self.learning_rate * grad
        w = w + self.momentum * self.momentum * (self.v - self.learning_rate * grad)
        return w
