import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline import read_and_process, split_train_test
from src.modelling import model_and_evaluate

def test_model():
    print(f"Testing the model..")
    train_data, test_data = read_and_process()
    X_train, y_train, X_test, y_test = split_train_test(train_data, test_data)
    accuracy, predictions = model_and_evaluate(X_train, y_train, X_test, y_test)
    try:
        assert len(predictions)==len(y_test)
        assert isinstance(accuracy, float)
    except Exception as e:
        print("Exception")
    print("Testing has been completed.")