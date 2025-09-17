import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import read_and_process, split_train_test
from src.modelling import model_and_evaluate

if __name__=="__main__":
    print(f"Executing this from {__name__}")
    train_data, test_data = read_and_process()
    X_train, y_train, X_test, y_test = split_train_test(train_data, test_data)
    accuracy, predictions = model_and_evaluate(X_train, y_train, X_test, y_test)
    print("Predictions have been made and accuracy has been calculated.")
    
    
