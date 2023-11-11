import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, load_breast_cancer

def load_california_housing(normalize=False):
    data = fetch_california_housing(as_frame=True).frame
    print(type(data))
    
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

if __name__ == '__main__':
    data, target = load_breast_cancer_data(normalize=True)
    # load_california_housing()
    # print(data.head())
    # print(target.head())