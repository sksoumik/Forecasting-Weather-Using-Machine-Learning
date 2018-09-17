import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

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

# Naive Bayes Gaussian Model setup
model = GaussianNB()
model.fit(x_train, y_train)

# Print results to evaluate model
print("Showing Performance Metrics for Naive Bayes Gaussian\n")

print ("Training Accuracy: {}".format(model.score(x_train, y_train)))
predicted = model.predict(x_test)
print ("Testing Accuracy: {}".format(accuracy_score(y_test, predicted)))

print("\n")

print("Cross Validation Accuracy: \n")
cv_accuracy = cross_val_score(estimator=model, X=x_train, y=y_train, cv=10)
print("Accuracy using 10 folds: ")
print(cv_accuracy)

print("\n")

print("Mean accuracy: {}".format(cv_accuracy.mean()))
print("Standard Deviation: {}".format(cv_accuracy.std()))

print("\n")

print("Confusion Matrix for Naive Bayes Gaussian\n")
labels = [0, 1, 2]
cm = confusion_matrix(y_test, predicted, labels=labels)
print(cm)

print("\n")

print('Precision, Recall and f-1 Scores for Naive Bayes Gaussian\n')
print(classification_report(y_test, predicted))
