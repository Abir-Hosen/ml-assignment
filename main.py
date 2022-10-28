import pandas as pd
data = pd.read_csv('dataset.csv')
print(data.head())
print(data.describe())
print(data.info())

data = data.drop(axis=1, columns=["sample_id"])

import matplotlib.pyplot as plt
# data.hist(bins=50, figsize=(20,15))
# plt.savefig('data-hist.png')
# plt.show()

print(data['variety'].value_counts())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
cat = data["variety"]
encoded = encoder.fit_transform(cat)
data["variety"] = encoded

from pandas.plotting import scatter_matrix
attributes = ["length", "width", "thickness", "surface_area", "mass", "compactness", "hardness", "shell_top_radius", "water_content", "carbohydrate", "variety"]
scatter_matrix(data[attributes], figsize=(20,15))
# plt.savefig('data-correlation.png')
# plt.show()

corr_matrix = data.corr()
print(corr_matrix['variety'].sort_values(ascending=False))

X = data.drop(columns=["variety"]).values
y = data["variety"].values

from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train,y_train)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='gini', n_estimators=50, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred_knn = knn.predict(X_test)
print("KNN accuracy on test set: %.3f" % accuracy_score(y_test, y_pred_knn))

y_pred_forest = forest.predict(X_test)
print("Random Forest accuracy on test set: %.3f" % accuracy_score(y_test, y_pred_forest))

from sklearn.model_selection import cross_val_score
import numpy as np
knn_cv_scores = cross_val_score(knn, X, y, cv=10)
print("KNN cross validation scores are: ", knn_cv_scores, "and scores mean:{}".format(np.mean(knn_cv_scores)))
forest_cv_scores = cross_val_score(forest, X, y, cv=10)
print("Random Forest cross validation Scores are: ", forest_cv_scores, "and scores mean:{}".format(np.mean(forest_cv_scores)))




