import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fawad_torch.nn import CrossEntropyLoss
import fawad_torch.optimizers as optim
from fawad_torch.nn import Linear, Sigmoid
from utils.datasets import load_breast_cancer_data
from sklearn.model_selection import train_test_split

class ClassificationNetwork:
    def __init__(self):
        self._layers = []
        self._optimizer = None

    def add_layer(self, node):
        self._layers.append(node)

    def add_optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, x, train=True):
        for layer in self._layers:
            x = layer.forward(x, train)
        return x
    
    def backward(self, error_tensor):
        for layer in self._layers[::-1]:
            error_tensor = layer.backward(error_tensor)
            if layer.is_trainable():
                if self._optimizer is None:
                    print("Must add optimizer using add_optimizer function")
                layer.w = self._optimizer.step(layer.w, layer.grad_weights)
        return error_tensor

x, y = load_breast_cancer_data(normalize=True)
x_train, x_val, y_train, y_val = train_test_split(x.to_numpy(), y.to_numpy(), test_size=0.3, random_state=42)
y_train = y_train.reshape((y_train.shape[0], 1))
y_val = y_val.reshape((y_val.shape[0], 1))

net = ClassificationNetwork()
net.add_layer(Linear(in_features=x_train.shape[1], out_features=256))
net.add_layer(Sigmoid())
# net.add_layer(Linear(in_features=256, out_features=256))
# net.add_layer(Sigmoid())
net.add_layer(Linear(in_features=256, out_features=y_train.shape[1]))
net.add_layer(Sigmoid())
net.add_optimizer(optim.SGD(learning_rate=1e-3))
loss_fn = CrossEntropyLoss(binary=True)
val_loss_fn = CrossEntropyLoss(binary=True)

epochs = 500
losses = []
val_losses = []
preds = []
for e in range(epochs):
    y_preds = net.forward(x_train)
    y_val_preds = net.forward(x_val, train=False)
    loss = loss_fn.forward(y_preds, y_train)
    val_loss = val_loss_fn.forward(y_val_preds, y_val)
    error_tensor = loss_fn.backward()
    grads = net.backward(error_tensor)
    losses.append(loss)
    val_losses.append(val_loss)

plt.figure()
plt.title("Training Loss")
plt.plot(range(len(losses)), losses)

plt.figure()
plt.title("Validation Loss")
plt.plot(range(len(val_losses)), val_losses)

plt.show()