import numpy as np
import matplotlib.pyplot as plt
from losses.mse_loss import MSELoss

class LinearRegression:
    def __init__(self):
        self.w = None
        self.losses = []

    def fit(self, x_train, y_train, x_val=None, y_val=None, learning_rate=0.001, epochs=100, batch_size=32, epsilon=1e-3, normal_eq=False):
        self.mean_x, self.std_x = x_train.mean(axis=0), x_train.std(axis=0)
        self.mean_y, self.std_y = y_train.mean(axis=0), y_train.std(axis=0)
        x_train = (x_train - self.mean_x) / self.std_x
        y_train = (y_train - self.mean_y) / self.std_y
        y_train = y_train.reshape((y_train.shape[0], 1))
        if y_val:
            y_val = y_val.reshape((y_val.shape[0], 1))
        if normal_eq:
            self.w = np.dot(np.linalg.pinv(np.dot(x_train.T, x_train)), np.dot(x_train.T, y_train))
        else:
            m, n = x_train.shape
            self.w = np.random.uniform(size=(n, 1))
            if m >= batch_size:
                iterations = m // batch_size + m % batch_size
            else:
                iterations = 1
            if (x_val is None and y_val is not None) or (x_val is not None and y_val is None):
                raise Exception("x_val and y_val need to be provided together")
            self.losses = []
            for e in range(epochs):
                index = 0
                for i in range(iterations - 1):
                    x_train_batch = x_train[index: index+batch_size]
                    y_train_batch = y_train[index: index+batch_size]
                    if y_train_batch.shape != (batch_size, 1):
                        y_train_batch = y_train_batch.reshape((batch_size, 1))
                    h_train_batch = np.dot(x_train_batch, self.w)
                    loss, grad = MSELoss(y_pred=h_train_batch, y_true=y_train_batch, train=True)
                    dw = np.dot(x_train_batch.T, grad)
                    self.w = self.w - learning_rate * dw
                    self.losses.append(loss)

    def plot(self):
        if len(self.losses) <= 0:
            print("Unable to plot losses as there are no losses recorded")
            return
        plt.plot(range(len(self.losses)), self.losses)
        plt.show()

    def predict(self, x):
        if self.w is None:
            print("Model not fit yet.")
            return
        return np.dot(x, self.w)