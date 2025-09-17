import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import read_and_process, split_train_test

import pandas as pd
import numpy as np
import sklearn
import tensorflow

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix,  roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, EarlyStopping


def model_and_evaluate(X_train, y_train, X_test, y_test):
    print()
    print("Gradient Boosting model has been chosen.")
    model = GradientBoostingClassifier(n_estimators=250, random_state=32)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # print("Predictions have been made.")
    accuracy = accuracy_score(y_test, predictions)*100
    print(f"Accuracy:{accuracy:.2f}%")
    return accuracy, predictions