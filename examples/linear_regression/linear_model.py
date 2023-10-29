import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None

    def fit(self, x_train, y_train, x_val=None, y_val=None, learning_rate=0.001, epochs=100, batch_size=32, epsilon=1e-3, normal_eq=False):
        y_train = y_train.reshape((y_train.shape[0], 1))
        if y_val:
            y_val = y_val.reshape((y_val.shape[0], 1))
        if normal_eq:
            self.w = np.dot(np.linalg.pinv(np.dot(x_train.T, x_train)), np.dot(x_train.T, y_train)) 
        m, n = x_train.shape
        # if m >= batch_size:
        #     iterations = m // batch_size + m % batch_size
        # else:
        #     iterations = 1
        # if (x_val is None and y_val is not None) or (x_val is not None and y_val is None):
        #     raise Exception("x_val and y_val need to be provided together")
        # for e in range(epochs):
        #     for i in range(iterations - 1):

    def predict(self, x):
        if self.w is None:
            print("Model not fit yet.")
            return
        return np.dot(x, self.w)