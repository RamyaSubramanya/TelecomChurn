import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import shutil
import kagglehub


def read_and_process():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    #define final data paths
    destination_train_path = os.path.join(data_dir, "churn_train.csv")
    destination_test_path = os.path.join(data_dir, "churn_test.csv")
    
    #download only if files dont exist locally
    if not (os.path.exists(destination_train_path) and os.path.exists(destination_test_path)):
        print("Downloading dataset from kagglehub.")
        try:
            downloaded_path = kagglehub.dataset_download("mnassrib/telecom-churn-datasets")
            if downloaded_path is None:
                raise FileNotFoundError("Downloaded path is empty, no data found.")
            
            #locate source files
            train_file_path = os.path.join(downloaded_path, 'churn-bigml-80.csv')
            test_file_path = os.path.join(downloaded_path, 'churn-bigml-20.csv')

            #copy to repo data folder
            shutil.copy(train_file_path, destination_train_path)
            shutil.copy(test_file_path, destination_test_path)
            print("Data downloaded and saved to 'data/' folder.")
        except Exception as e:
            return f"Error while downloading data:{e}"
    else:
        print("Using existing dataset from data folder.")
        
    #load into pandas
    train_data = pd.read_csv(destination_train_path)
    test_data = pd.read_csv(destination_test_path)
        
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