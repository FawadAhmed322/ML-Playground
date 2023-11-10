import numpy as np
import matplotlib.pyplot as plt
from fawad_torch.nn import Linear
from losses.mse_loss import MSELoss
import fawad_torch.optimizers as optim

class LinearRegression:
    def __init__(self):
        self.linear_layer = None
        self.loss = MSELoss()
        self.optimizer = None
        self.losses = []
        self.val_losses = []
        self.mean_x, self.mean_y = None, None

    def step(self, gen, train=True):
        try:
            x_batch, y_batch = next(gen)
            y_preds_batch = self.linear_layer.forward(x_batch)
            loss = self.loss.forward(y_preds_batch, y_batch)
            grad = None
            if train:
                loss_grad = self.loss.backward()
                grad = self.linear_layer.backward(loss_grad)
            return loss, grad
        except:
            raise Exception()
    
    def normalize_data(self, x, y, validation=False):
        if validation:
            if self.mean_x is None or self.mean_y is None:
                raise Exception("Training data must be normalized first to store mean and standard deviation")
        elif not validation:
            self.mean_x, self.std_x = x.mean(axis=0), x.std(axis=0)
            self.mean_y, self.std_y = y.mean(axis=0), y.std(axis=0)
        x_norm = (x - self.mean_x) / (self.std_x + np.finfo(float).eps)
        y_norm = (y - self.mean_y) / (self.std_y + np.finfo(float).eps)
        return (x_norm, y_norm)
        
    
    def batch_generator(self, x, y, batch_size):
        if batch_size is None:
            yield x, y
            return
        index = 0
        while index + batch_size <= x.shape[0]:
            batch = (x[index: index+batch_size], y[index: index+batch_size])
            index += batch_size
            yield batch
        batch = (x[index: index + x.shape[0] % batch_size], y[index: index + y.shape[0] % batch_size])
        yield batch

    def fit(self, x_train, y_train, x_val=None, y_val=None, learning_rate=1e-9, epochs=300, batch_size=32, normal_eq=False, bias=True):
        self.bias = bias
        val = False
        if (x_val is None and y_val is not None) or (x_val is not None and y_val is None):
                raise Exception("x_val and y_val need to be provided together")
        elif not (x_val is None or y_val is None):
            val = True
        self.val = val
        x_train, y_train = self.normalize_data(x_train, y_train)
        if val:
            x_val, y_val = self.normalize_data(x_val, y_val, validation=True)
        if normal_eq:
            if bias:
                x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
            self.linear_layer = Linear(in_features=x_train.shape[1], out_features=y_train.shape[1], bias=bias)
            self.linear_layer.w = np.dot(np.linalg.pinv(np.dot(x_train.T, x_train)), np.dot(x_train.T, y_train))
        else:
            self.linear_layer = Linear(in_features=x_train.shape[1], out_features=y_train.shape[1], bias=bias)
            self.optimizer = optim.SGD(learning_rate)
            self.losses = []
            if val:
                self.val_losses = []
            for e in range(epochs):
                train_gen = self.batch_generator(x_train, y_train, batch_size)
                losses = []
                while True:
                    try:
                        loss, grad = self.step(train_gen)
                        loss = loss[0]
                        if not np.isnan(loss):
                            self.linear_layer.w = self.optimizer.step(self.linear_layer.w, grad)
                            losses.append(loss)
                        if val:
                            val_gen = self.batch_generator(x_val, y_val, batch_size)
                            val_losses = []
                            while True:
                                try:
                                    val_loss, _ = self.step(val_gen)
                                    val_loss = val_loss[0]
                                    if not np.isnan(val_loss):
                                        val_losses.append(val_loss)
                                except:
                                    break
                            avg_val_loss = np.array(val_losses).mean()
                            self.val_losses.append(avg_val_loss)
                    except:
                        break
                avg_loss = np.array(losses).mean()
                self.losses.append(avg_loss)
                print(f"Epoch: {e+1}/{epochs}\nTraining Loss: {self.losses[-1]}")
                if val:
                    print(f"Validation Loss: {self.val_losses[-1]}\n")
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
        if self.linear_layer.w is None:
            print("Model not fit yet.")
            return
        return self.linear_layer.forward(x)