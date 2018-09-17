import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# Read dataset from csv
dataset = pd.read_csv("dataset.csv")
print ("Total number of rows in dataset: {}\n".format(len(dataset)))
print(dataset.head())

# Features
features = ['Day','Month','Year','Humidity','Max Temperature','Min Temperature',
            'Rainfall','Sea Level Pressure','Sunshine','Wind Speed']
target = 'Cloud'

x_train, x_test, y_train, y_test = train_test_split(dataset[features], dataset[target],
                                                    train_size=0.7, test_size=0.3, shuffle=False)

# Print samples after running train_test_split
print("X_train: {}, Y_train: {}".format(len(x_train), len(x_test)))
print("X_train: {}, Y_train: {}".format(len(y_train), len(y_test)))

print("\n")

# Support Vector Machine model parameter tuning
model = svm.SVC()

param_grid = [{'kernel': ['rbf'],
               'gamma': [1e-4, 0.01, 0.1],
               'C': [0.01, 1]}]

print("Hyper Parameter Tuning Results\n")

# Finding optimum parameters through GridSearchCV
grid = GridSearchCV(estimator=model, param_grid = param_grid,
                    cv = 5)
grid.fit(x_train, y_train)

print("\n")
print("Results returned by GridSearchCV\n")
print("Best estimator: ", grid.best_estimator_)
print("\n")
print("Best score: ", grid.best_score_)
print("\n")
print("Best parameters found: ", grid.best_params_)