import os
import pandas as pd
data = pd.read_csv('dataset.csv')
print(data.head())
print(data.info())
print(data["variety"].value_counts())
print(data.describe())

import matplotlib.pyplot as plt
# data.hist(bins=50, figsize=(20,15))
# plt.show()

import numpy as np

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

print(train_set.info())

# data.plot(kind='scatter', x='width', y='thickness')
# data.plot(kind='scatter', x='width', y='length')
# data.plot(kind='scatter', x='length', y='thickness')
# data.plot(kind='scatter', x='surface_area', y='thickness')
# data.plot(kind='scatter', x='surface_area', y='compactness')
# data.plot(kind='scatter', x='hardness', y='compactness')
print(data["carbohydrate"].value_counts())

from pandas.plotting import scatter_matrix
# attributes = ["length", "width", "thickness", "surface_area", "mass", "compactness", "hardness", "shell_top_radius", "water_content", "carbohydrate", "variety"]
# scatter_matrix(data[attributes], figsize=(12,8))
# plt.show()

corr_matrix = data.corr()
print(corr_matrix["length"].sort_values(ascending=False))

new_data = train_set.drop("variety", axis=1)
new_data_label = train_set.dropna["variety"]

print(new_data.head())