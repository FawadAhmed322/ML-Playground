from optimizers.base import BaseOptimizer as _BaseOptimizer

class SGD(_BaseOptimizer):
    """Stochastic Gradient Descent"""
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def step(self, w, grad):
        w = w - self.learning_rate * grad
        return w
    
if __name__ == '__main__':
    x = SGD(42)
    print(x.learning_rate)