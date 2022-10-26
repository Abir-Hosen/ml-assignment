import pandas as pd
df = pd.read_csv("dataset.csv").drop(axis=1, columns=["sample_id"])
df.head()
df.shape

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
cat = df["variety"]
encoded = encoder.fit_transform(cat)
df["variety"] = encoded



#create a dataframe with all training data except the target column
X = df.drop(columns=["variety"])

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
('std_scaler', StandardScaler()),
])
X = pd.DataFrame(num_pipeline.fit_transform(X))

#check that the target variable has been removed
X.head()

y = df["variety"].values
#view target values
y[0:]


from sklearn.model_selection import train_test_split
#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)



from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
# Fit the classifier to the data
knn.fit(X_train,y_train)



#show first 5 model predictions on the test data
print(knn.predict(X_test)[0:5], y_test[0:5])



#check accuracy of our model on the test data
print(knn.score(X_test, y_test))



from sklearn.model_selection import cross_val_score
import numpy as np
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=5)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, X, y, cv=10)
#print each cv score (accuracy) and average them
print(cv_scores)
print("cv_scores mean:{}".format(np.mean(cv_scores)))



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

forest = RandomForestClassifier(criterion='gini',
                                 n_estimators=50,
                                 random_state=1,
                                 n_jobs=2)

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)
print("Accuracy: %.3f" % accuracy_score(y_test, y_pred))


rcv_scores = cross_val_score(forest, X, y, cv=10)
#print each cv score (accuracy) and average them
print(rcv_scores)
print("rcv_scores mean:{}".format(np.mean(rcv_scores)))

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

 
#
# plot_decision_regions function takes "forest" as classifier
#
dicct =  {1: "length", 2:"width", 3:"thickness", 4:"surface_area", 5:"mass", 6:"compactness", 7:"hardness", 8:"shell_top_radius"}#, 9:"water_content", 10:"carbohydrate", 11:"variety"}

values = 1
width = 5
filler_feature_values={ 0:-1, 1:2, 2:values, 3:values, 4:values, 5:values, 6:values, 7:values, 8:values, 9:values}
filler_feature_ranges={ 0:width, 1:width, 2:width, 3:width, 4:width, 5:width, 6:width, 7:width, 8:width, 9:width}

fig, ax = plt.subplots(figsize=(7, 7))

plot_decision_regions(X_combined, y_combined, clf=forest, feature_index=[0,1], filler_feature_values=filler_feature_values, filler_feature_ranges=filler_feature_ranges, legend=3, ax=ax)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


