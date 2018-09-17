import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

DATA_SET_PATH = "dataset.csv"
dataset = pd.read_csv(DATA_SET_PATH)
print ("Number of samples in dataset:", len(dataset), "\n")
print(dataset.head())

features = ['Day','Month','Year','Humidity','Max Temperature','Min Temperature',
            'Rainfall','Sea Level Pressure','Sunshine','Wind Speed']
target = 'Cloud'

x_train, x_test, y_train, y_test = train_test_split(dataset[features], dataset[target],
                                                    train_size=0.7, test_size=0.3, shuffle=False)

# Print samples after running train_test_split
print("X_train: {}, Y_train: {}".format(len(x_train), len(x_test)))
print("X_train: {}, Y_train: {}".format(len(y_train), len(y_test)))

print("\n")

# Multinomial Naive Bayes model parameter tuning
model = MultinomialNB()
param_grid = {'alpha': [1.0, 2.0, 3.0, 4.0, 5.0]}

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