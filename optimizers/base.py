class BaseOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, w, grad):
        pass