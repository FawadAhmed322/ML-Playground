import pandas as pd
import numpy as np
from fawad_utils.utils import indices_to_one_hot
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_iris, load_digits

def load_california_housing(normalize=False):
    data = fetch_california_housing(as_frame=True).frame
    if normalize:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    x = data.drop(columns=['MedHouseVal'])
    y = data['MedHouseVal']
    return x, y

def load_breast_cancer_data(normalize=False):
    data, target = load_breast_cancer(return_X_y=True, as_frame=True)
    if normalize:
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data, target

def load_iris_data(normalize=False):
    data_bunch = load_iris()
    data, target, target_names = data_bunch['data'], data_bunch['target'], data_bunch['target_names']
    target_ones = indices_to_one_hot(indices=target, n_classes=len(target_names))
    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    return data, target_ones

def load_mnist_data(normalize=False):
    data_bunch = load_digits()
    data, target, target_names = data_bunch['data'], data_bunch['target'], data_bunch['target_names']
    target_ones = indices_to_one_hot(indices=target, n_classes=len(target_names))
    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    return data, target_ones

if __name__ == '__main__':
    load_mnist_data()
    # data, target = load_iris_data()
    # print(data.head())
    # print(target.head())
    
    # load_california_housing()
    # print(data.head())
    # print(target.head())