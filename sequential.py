# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:22:51 2023

@author: USER
"""

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


import time

# Record the start time
start_time = time.time()

data = pd.read_csv('C:/Users/USER/Downloads/Iris.csv')

X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=5)
# Define a function to train and predict for a given classifier
def train_and_predict(classifier, X_train, y_train, X_test):
    start_time = time.time()
    clf.fit(X_train, y_train)
    end_time = time.time()
    y_pred = clf.predict(X_test)
    run_time = end_time - start_time
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return y_pred, accuracy, run_time

# Define classifiers
classifiers = [
    KNeighborsClassifier(n_neighbors=12),
    LogisticRegression(),
    SVC(kernel='linear')
]

# Loop through classifiers
for clf in classifiers:
    clf_name = clf.__class__.__name__
    # Train and predict
    y_pred, accuracy, run_time= train_and_predict(clf, X_train, y_train, X_test)
    classification_rep = metrics.classification_report(y_test, y_pred)
    print(f"Execution time: {run_time} seconds")
    # Print results
    print(f"Classifier: {clf_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{classification_rep}\n")
 
    cm = metrics.confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data['Species'].unique(), yticklabels=data['Species'].unique())
    plt.title(f'Confusion Matrix - {clf_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{clf_name}.png')  # Save the figure as an image
    plt.show()

    print("\n")
# Record the end time
end_time = time.time()
# Calculate the total run time
total_run_time = end_time - start_time
print(f"Total execution time for all algorithms sequntial: {total_run_time} seconds.")
