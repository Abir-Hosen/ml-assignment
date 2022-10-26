import pandas as pd
df = pd.read_csv("dataset.csv").drop(axis=1, columns=["sample_id"])
df.head()
df.shape

#create a dataframe with all training data except the target column
X = df.drop(columns=["variety"])
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