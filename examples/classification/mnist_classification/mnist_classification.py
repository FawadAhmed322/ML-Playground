import matplotlib.pyplot as plt
import fawad_torch.optimizers as optim
from fawad_torch.nn import Linear, ReLU, Softmax, CrossEntropyLoss
from fawad_utils.datasets import load_mnist_data
from sklearn.model_selection import train_test_split

class ClassificationNetwork:
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

x, y = load_mnist_data(normalize=True)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

net = ClassificationNetwork()
net.add_layer(Linear(in_features=x_train.shape[1], out_features=32, name='Linear 1'))
net.add_layer(ReLU(name='Relu 1'))
net.add_layer(Linear(in_features=32, out_features=y_train.shape[1], name='Linear 2'))
net.add_layer(Softmax(name='Softmax 1'))
net.add_optimizer(optim.Adam)

loss_fn = CrossEntropyLoss()
val_loss_fn = CrossEntropyLoss()

epochs = 1000
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
    print(loss)

plt.figure()
plt.title("Training Loss")
plt.plot(range(len(losses)), losses)
plt.savefig("examples/classification/mnist_classification/training_loss.png")

plt.figure()
plt.title("Validation Loss")
plt.plot(range(len(val_losses)), val_losses)
plt.savefig("examples/classification/mnist_classification/validation_loss.png")

plt.show()