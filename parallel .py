# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:00:22 2023

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
from sklearn.metrics import confusion_matrix


import time
from joblib import Parallel, delayed
# Record the start time
start_time = time.time()
data = pd.read_csv('C:/Users/USER/Downloads/Iris.csv')
# Modeling with scikit-learn
X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=5)
def train_and_predict(clf, X_train, y_train, X_test):
    """
    Trains a classifier and predicts the labels for the test data.

    Args:
        clf: A scikit-learn classifier.
        X_train: The training data.
        y_train: The training labels.
        X_test: The test data.

    Returns:
        y_pred: The predicted labels for the test data.
        accuracy: The accuracy of the classifier.
        run_time: The time it took to train the classifier.
    """

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

# Define a function to calculate and display the confusion matrix
def calculate_confusion_matrix(y_true, y_pred, classifier_name):
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix for {classifier_name}:")
    print(cm)
# Loop through classifiers
for clf in classifiers:
    clf_name = clf.__class__.__name__

    # Using 'loky' backend
    with Parallel(n_jobs=2, backend='loky'):
        y_pred, accuracy, run_time = train_and_predict(clf, X_train, y_train, X_test)
        classification_rep = metrics.classification_report(y_test, y_pred)
        # Print accuracy
        print(f"{clf_name} - Loky Backend: {run_time} seconds.")
        print(f"Accuracy: {accuracy}")
        calculate_confusion_matrix(y_test, y_pred, clf_name)
        # Generate classification report
        classification_rep = metrics.classification_report(y_test, y_pred)
        print(f"Classification Report:\n{classification_rep}")
        
        
    with Parallel(n_jobs=2, backend='threading'):
        y_pred, accuracy, run_time = train_and_predict(clf, X_train, y_train, X_test)
        # Print accuracy
        print(f"{clf_name} - Threading Backend: {run_time} seconds.")
        print(f"Accuracy: {accuracy}")
        # Generate classification report
        classification_rep = metrics.classification_report(y_test, y_pred)
        print(f"Classification Report:\n{classification_rep}")
      

    with Parallel(n_jobs=2, backend='multiprocessing'):
        y_pred, accuracy, run_time = train_and_predict(clf, X_train, y_train, X_test)
        # Print accuracy
        print(f"{clf_name} - Multiprocessing Backend: {run_time} seconds.")
        print(f"Accuracy: {accuracy}")
        # Generate classification report
        classification_rep = metrics.classification_report(y_test, y_pred)
        print(f"Classification Report:\n{classification_rep}")           
end_time = time.time()
total_run_time = end_time - start_time
print(f"Total execution time for all algorithms: {total_run_time} seconds.")


