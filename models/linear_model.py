import numpy as np
import matplotlib.pyplot as plt
from losses.mse_loss import MSELoss

class LinearRegression:
    def __init__(self):
        self.w = None
        self.losses = []
        self.val_losses = []

    def forward_on_batch(self, x, y, train=False):
        if y.shape != (y.shape[0], 1):
            y_batch = y_batch.reshape((y.shape[0], 1))
        h_batch = np.dot(x, self.w)
        loss, grad = MSELoss(y_pred=h_batch, y_true=y, train=train)
        return loss, grad

    def fit(self, x_train, y_train, x_val=None, y_val=None, learning_rate=1e-6, epochs=300, batch_size=32, normal_eq=False, bias=True):
        self.bias = bias
        if bias:
            x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
        self.mean_x, self.std_x = x_train.mean(axis=0), x_train.std(axis=0)
        self.mean_y, self.std_y = y_train.mean(axis=0), y_train.std(axis=0)
        x_train = (x_train - self.mean_x) / (self.std_x + np.finfo(float).eps)
        y_train = (y_train - self.mean_y) / (self.std_y + np.finfo(float).eps)
        y_train = y_train.reshape((y_train.shape[0], 1))
        val = False
        if (x_val is None and y_val is not None) or (x_val is not None and y_val is None):
                raise Exception("x_val and y_val need to be provided together")
        elif not (x_val is None or y_val is None):
            val = True
        self.val = val
        if val:
            if bias:
                x_val = np.hstack([np.ones((x_val.shape[0], 1)), x_val])
            y_val = y_val.reshape((y_val.shape[0], 1))
            x_val = (x_val - self.mean_x) / (self.std_x + np.finfo(float).eps)
            y_val = (y_val - self.mean_y) / (self.std_y + np.finfo(float).eps)
        if normal_eq:
            self.w = np.dot(np.linalg.pinv(np.dot(x_train.T, x_train)), np.dot(x_train.T, y_train))
        else:
            m, n = x_train.shape
            self.w = np.random.uniform(size=(x_train.shape[1], 1))
            if m >= batch_size:
                iterations = m // batch_size + m % batch_size
            else:
                iterations = 1
            if val:
                m_val = x_val.shape[0]
                val_loss_batch = np.zeros(shape=(m_val))
                if m_val >= batch_size:
                    val_iterations = m_val // batch_size + m_val % batch_size
                else:
                    val_iterations = 1
            self.losses = []
            for e in range(epochs):
                index = 0
                losses = []
                val_losses = []
                for i in range(iterations - 1):
                    x_train_batch = x_train[index: index+batch_size]
                    y_train_batch = y_train[index: index+batch_size]
                    loss, grad = self.forward_on_batch(x_train_batch, y_train_batch, train=True)
                    dw = np.dot(x_train_batch.T, grad)
                    self.w = self.w - learning_rate * dw
                    losses.append(loss)
                    index += batch_size
                x_train_batch = x_train[index: index + m % batch_size]
                y_train_batch = y_train[index: index + m % batch_size]
                loss, grad = self.forward_on_batch(x_train_batch, y_train_batch, train=True)
                dw = np.dot(x_train_batch.T, grad)
                self.w = self.w - learning_rate * dw
                losses.append(loss)
                if val:
                    index = 0
                    for i in range(val_iterations - 1):
                        x_val_batch = x_train[index: index+batch_size]
                        y_val_batch = y_train[index: index+batch_size]
                        loss, grad = self.forward_on_batch(x_val_batch, y_val_batch, train=False)
                        val_losses.append(loss)
                        index += batch_size
                    x_val_batch = x_train[index: index + m_val % batch_size]
                    y_val_batch = y_train[index: index + m_val % batch_size]
                    loss, grad = self.forward_on_batch(x_val_batch, y_val_batch, train=False)
                    val_losses.append(loss)
                self.losses.append(np.array(losses).mean())
                self.val_losses.append(np.array(val_losses).mean())
        print(f"Final Loss: {self.losses[-1]}")
        if val:
            print(f"Final Validation Loss: {self.val_losses[-1]}")

    def plot(self):
        if len(self.losses) <= 0:
            print("Unable to plot losses as there are no losses recorded")
        else:
            plt.figure()  # Create a new figure
            plt.plot(range(len(self.losses)), self.losses)
            plt.title("Training Loss")
            plt.savefig("examples/linear_regression/training_loss.png")
        if self.val:
            if len(self.val_losses) <= 0:
                print("Unable to plot losses as there are no losses recorded")
            else:
                plt.figure()
                plt.plot(range(len(self.val_losses)), self.val_losses)
                plt.title("Validation Loss")
                plt.savefig("examples/linear_regression/validation_loss.png")
        plt.show()

    def predict(self, x):
        if self.w is None:
            print("Model not fit yet.")
            return
        return np.dot(x, self.w)