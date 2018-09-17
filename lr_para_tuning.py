import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
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

# Logistic Regression model parameter tuning
# Problem with converging, so manual tuning used
model = LogisticRegression()

param_grid = [{'penalty': ['l1'],
               'solver': ['liblinear', 'saga'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
              {'penalty': ['l2'],
               'solver': ['newton-cg', 'lbfgs', 'sag'],
               'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
               'max_iter': [100, 200, 300]}]


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