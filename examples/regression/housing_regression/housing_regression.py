import matplotlib.pyplot as plt
from losses.mse_loss import MSELoss
import fawad_torch.optimizers as optim
from fawad_torch.nn import Linear, ReLU
from fawad_utils.datasets import load_california_housing
from sklearn.model_selection import train_test_split

class RegressionNetwork:
    def __init__(self):
        self._layers = []

    def add_layer(self, node):
        self._layers.append(node)

    def add_optimizer(self, optimizer):
        for layer in self._layers:
            if layer.is_trainable():
                layer.add_optimizer(optimizer(learning_rate=1e-3))

    def forward(self, x, train=True):
        for layer in self._layers:
            x = layer.forward(x, train)
        return x
    
    def backward(self, error_tensor):
        for layer in self._layers[::-1]:
            error_tensor = layer.backward(error_tensor)
        return error_tensor

x, y = load_california_housing(normalize=True)
x_train, x_val, y_train, y_val = train_test_split(x.to_numpy(), y.to_numpy(), test_size=0.3, random_state=42)
y_train = y_train.reshape((y_train.shape[0], 1))
y_val = y_val.reshape((y_val.shape[0], 1))

net = RegressionNetwork()
net.add_layer(Linear(in_features=x_train.shape[1], out_features=64))
net.add_layer(ReLU())
net.add_layer(Linear(in_features=64, out_features=y_train.shape[1]))
net.add_optimizer(optim.NAG)
loss_fn = MSELoss()
val_loss_fn = MSELoss()

epochs = 120
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
plt.savefig("examples/regression/housing_regression/training_loss.png")

plt.figure()
plt.title("Validation Loss")
plt.plot(range(len(val_losses)), val_losses)
plt.savefig("examples/regression/housing_regression/validation_loss.png")

plt.show()