import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Read dataset from csv
dataset = pd.read_csv("dataset.csv")
print ("Total number of rows in dataset: {}\n".format(len(dataset)))
print(dataset.head())

features = ['Day','Month','Year','Humidity','Max Temperature','Min Temperature',
            'Rainfall','Sea Level Pressure','Sunshine','Wind Speed']
target = 'Cloud'

# Splitting X features and y target from dataset
x = dataset[features]
y = dataset[target]
y = label_binarize(y, classes=[0,1,2])
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, shuffle=False)

# Model returned after parameter tuning, not used because problem with convergence
# Manual tuned parameters used

# model = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=10000, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='newton-cg', tol=0.0001,
#           verbose=0, warm_start=False)

model = LogisticRegression(C=0.01, penalty='l1', solver='liblinear')

# Attempted ROC curve evaluation for the models, but not used because it does not work
# for multi class classifcation

# Working for binary class classification, but cloud states were divided into 3 states - dipanzan
# ROC curve and AUC value calculation
print("ROC curve and AUC value")
classifier = OneVsRestClassifier(estimator=model)
y_score = classifier.fit(x_train, y_train).decision_function(x_test)
fpr = []
tpr = []
roc_auc = []

# y_test = y_test[50:100]
# print(y_test)
# print(y_test[:, 0])
# print(y_test[:, 1])
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(roc_auc[i])

'''
predictions = model.predict_proba(x_test)

fpr, tpr, thresholds = roc_curve(y_test, predictions[:,1])
roc_auc = auc(fpr, tpr)
print ("AUC value of Logistic Regression model:", roc_auc)

plt.title('ROC Curve of Logistic Regression model')
plt.plot(fpr, tpr, color='navy', label='ROC of LR (AUC=%0.3f)' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate (TPR)')
plt.xlabel('False Positive Rate (FPR)')
#plt.show()
plt.savefig('ROC curve of Logistic Regression model.png')
'''