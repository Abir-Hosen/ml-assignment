import os
import pandas as pd
data = pd.read_csv('dataset.csv')
# print(data.head())
# print(data.info())
# print(data["variety"].value_counts())
# print(data.describe())

import matplotlib.pyplot as plt
# data.hist(bins=50, figsize=(20,15))
# plt.savefig('data-hist.png')
# plt.show()

import numpy as np

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# print(train_set.info())

# data.plot(kind='scatter', x='width', y='thickness')
# data.plot(kind='scatter', x='width', y='length')
# data.plot(kind='scatter', x='length', y='thickness')
# data.plot(kind='scatter', x='surface_area', y='thickness')
# data.plot(kind='scatter', x='surface_area', y='compactness')
# data.plot(kind='scatter', x='hardness', y='compactness')
# print(data["carbohydrate"].value_counts())

from pandas.plotting import scatter_matrix
# attributes = ["length", "width", "thickness", "surface_area", "mass", "compactness", "hardness", "shell_top_radius", "water_content", "carbohydrate", "variety"]
# scatter_matrix(data[attributes], figsize=(12,8))
# plt.show()

corr_matrix = data.corr()
# print(corr_matrix["length"].sort_values(ascending=False))

new_data = train_set.drop(axis=1, columns=["variety","sample_id"])
new_data_label = train_set["variety"].copy()

new_data_test = test_set.drop(axis=1, columns=["variety","sample_id"])
new_data_test_label = test_set.drop(axis=1, columns=["sample_id"])["variety"].copy()

# print(new_data.info())


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
('std_scaler', StandardScaler()),
])
new_data_tr = num_pipeline.fit_transform(new_data)
new_data_tst = num_pipeline.fit_transform(new_data_test)

# print(new_data_tr.shape)
print(new_data_tst[4])
# print( '\n----------------------------------------------------     ','\n','\n',new_data_test_label.info(),'\n','\n')#, '\n', new_data_label.head())
print( '\n----------------------------------------------------     \n', new_data_label.head())

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(new_data_tr, new_data_label)

print(knn_clf.predict([new_data_tst[4]]))


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print(new_data_test)

# for train_index, test_index in skfolds.split(new_data_tr, new_data_label):
#     # print('train index', test_index)
#     clone_clf = clone(knn_clf)
#     X_train_folds = new_data_tr[train_index]
#     y_train_folds = new_data_label[train_index]
#     X_test_fold = new_data_tr[test_index]
#     y_test_fold = new_data_label[test_index]
#     clone_clf.fit(X_train_folds, y_train_folds)
#     y_pred = clone_clf.predict(X_test_fold)
#     # n_correct = sum(y_pred == y_test_fold)
#     # print(n_correct / len(y_pred)) # prints 0.9502, 0.96565 and 0.96495

