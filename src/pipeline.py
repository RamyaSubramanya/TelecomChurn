import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import shutil
import kagglehub


def read_and_process():
    try:
    # Download latest version
        downloaded_path = kagglehub.dataset_download("mnassrib/telecom-churn-datasets")
        if downloaded_path is None:
            print("Downloaded path is empty, no data found.")
    except Exception as e:
        return f"Error:{e}"
        
    #get current working directory to save dataset
    data_path = "D:\Data Science\Machine Learning & Deep Learning ANN (Regression & Classification)\Classification Practicals\TelecomChurn\data"
    print("Data has been loaded from Kagglehub.")
    
    #get train and test data paths
    train_file_path = os.path.join(downloaded_path, 'churn-bigml-80.csv')
    test_file_path = os.path.join(downloaded_path, 'churn-bigml-20.csv')

    destination_train_path = os.path.join(data_path, "churn_train.csv")
    destination_test_path = os.path.join(data_path, "churn_test.csv")

    #save train, test data files
    train_data = pd.read_csv(shutil.copy(train_file_path, destination_train_path))
    test_data = pd.read_csv(shutil.copy(test_file_path, destination_test_path))
    
    print("Data pre-processing begins...")
    
    #select the relevant columns
    train_data = train_data[['Account length', 'International plan', 'Voice mail plan', 'Number vmail messages', 
                         'Total day minutes', 'Total day calls', 'Total day charge', 'Total eve minutes',
                         'Total eve calls', 'Total eve charge', 'Total night minutes', 'Total night calls', 
                         'Total night charge', 'Total intl minutes', 'Total intl calls', 'Total intl charge', 
                         'Customer service calls', 'Churn']]

    #convert categorical into dummies
    train_data = pd.get_dummies(train_data, columns=['International plan', 'Voice mail plan'], 
                                drop_first=True, dtype='int64')

    #convert target boolean into integer encoding
    train_data['Churn'] = train_data['Churn'].map({False:0, True:1})

    test_data = test_data[['Account length', 'International plan', 'Voice mail plan', 'Number vmail messages', 
                            'Total day minutes', 'Total day calls', 'Total day charge', 'Total eve minutes',
                            'Total eve calls', 'Total eve charge', 'Total night minutes', 'Total night calls', 
                            'Total night charge', 'Total intl minutes', 'Total intl calls', 'Total intl charge', 
                            'Customer service calls', 'Churn']]

    test_data = pd.get_dummies(test_data, columns=['International plan', 'Voice mail plan'], 
                                drop_first=True, dtype='int64')

    test_data['Churn'] = test_data['Churn'].map({False:0, True:1})
    print("Train and test data have been preprocessed and saved sucessfully.")
    return train_data, test_data

def split_train_test(train_data, test_data):
    #split the independent and target variable
    X_train = train_data.drop(columns=['Churn'])
    y_train = train_data['Churn']

    X_test = test_data.drop(columns=['Churn'])
    y_test = test_data['Churn']
    print("Train and test data have been split into X_train, X_test, y_train and y_test for modelling.")
    return X_train, y_train, X_test, y_test