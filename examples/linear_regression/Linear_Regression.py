import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from models.linear_model import LinearRegression
from losses.mse_loss import MSELoss

data = fetch_california_housing(as_frame=True).frame
x = data.drop(columns=['MedHouseVal'])
y = data['MedHouseVal']
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(x_train.values, y_train.values, x_val=x_val.values, y_val=y_val.values)
reg.plot()