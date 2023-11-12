from activations.base import BaseActivation
import numpy as np

class Softmax(BaseActivation):
    def __init__(self, name=None, loops=False):
        super().__init__(name=name)
        self.loops = loops

    def forward(self, z, train=True):
        z_exp = np.exp(z - np.max(z, axis=-1, keepdims=True))
        sum_exp = np.sum(z_exp, axis=-1, keepdims=True)
        out = z_exp / sum_exp
        if train:
            self.out = out
        return out
    
    def backward(self, error_tensor):
        if not self.loops:
            # Calculate the gradient of the softmax
            jacobian = np.zeros((self.out.shape[0], self.out.shape[1], self.out.shape[1]))
            diag_indices = np.diag_indices(self.out.shape[1])
            jacobian[:, diag_indices[0], diag_indices[1]] = self.out * (1 - self.out)
            off_diag_indices = np.where(~np.eye(self.out.shape[1], dtype=bool))
            jacobian[:, off_diag_indices[0], off_diag_indices[1]] = -self.out[:, off_diag_indices[0]] * self.out[:, off_diag_indices[1]]
            grad = np.einsum('ijk,ik->ij', jacobian, error_tensor)
        else:
            jacobian = np.zeros((self.out.shape[0], self.out.shape[1], self.out.shape[1]))
            for i in range(self.out.shape[0]):
                for j in range(self.out.shape[1]):
                    for k in range(self.out.shape[1]):
                        if j == k:
                            jacobian[i][j][k] = self.out[i][j] * (1 - self.out[i][j])
                        else:
                            jacobian[i][j][k] = -self.out[i][j] * self.out[i][k]
            grad = np.einsum('ijk,ik->ij', jacobian, error_tensor)
        return grad