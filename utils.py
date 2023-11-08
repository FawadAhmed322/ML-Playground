import numpy as np

def indices_to_one_hot(indices, n_classes):
    one_hot_matrix = np.zeros((len(indices), n_classes))
    one_hot_matrix[np.arange(len(indices)), indices] = 1
    return one_hot_matrix