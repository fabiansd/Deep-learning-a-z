# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_country = LabelEncoder()
X[:, 1] = labelencoder_country.fit_transform(X[:, 1])
labelencoder_gender = LabelEncoder()
X[:, 2] = labelencoder_gender.fit_transform(X[:, 2])

#Variables are one-hot-encoded
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#We remove the forst one-hot variable to avoid the "one-hod trap"
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

def build_classifier(optimizer_choice = 'adam', dropout_rate = 0.2):
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init='uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(p = dropout_rate))
    classifier.add(Dense(output_dim = 6, init='uniform', activation = 'relu'))
    classifier.add(Dropout(p = dropout_rate))
    classifier.add(Dense(output_dim = 1, init='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer_choice, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# K-fold crossvalidation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# # use Keras wrapper that merges scikit and keras
# classifier_obj = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 20)

# # Train the classifier in a k-fold fashion
# accuracies = cross_val_score(estimator=classifier_obj, X=X_train, y=y_train,  cv=10, n_jobs=1)

# mean = accuracies.mean()
# variance = accuracies.std()

# print('\n Mean acc: {} std: {}'.format(mean, variance))
# print('Accuracies: {}'.format(accuracies))

# Gridsearch for tuning network
from sklearn.model_selection import GridSearchCV

#remake the classifier w/o params specified
classifier_obj = KerasClassifier(build_fn = build_classifier)

# Create a dict with different parameter options
grid_search_parameters = {'batch_size' : [16,32], 'epochs' : [30,50], 'optimizer_choice': ['adam', 'rmsprop'], 'dropout_rate' : [0.1,0.2,0.3]}
grid_search_obj = GridSearchCV(classifier_obj, param_grid=grid_search_parameters, scoring='accuracy',cv=10)

# Fit the grid search. This will take several hours
grid_search_obj.fit(X_train, y_train)

# Will find the best parameter selection from the grid search dictionary
best_parameters = grid_search_obj.best_params_
best_accuracy = grid_search_obj.best_score_

# #Predicting the Test set results
# y_pred = classifier_obj.predict(X_test)
# y_pred = (y_pred > 0.5)

# # # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)

# print('Confusion matrix:')
# print(cm)


# with open('cross var result.txt', 'wb') as file:
#     for item in accuracies:
#         file.write(' {} '.format(item))
#     file.write('\n Mean acc: {} std: {}\n\n'.format(mean, variance))