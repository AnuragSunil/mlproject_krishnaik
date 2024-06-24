import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from src.exception import CustomException

def evaluate_models(X_train, y_train,X_test,y_test, models):
    """
    Evaluates multiple regression models on the given dataset and returns a report
    with the R2 score for the test set of each model.

    Parameters:
    X (pd.DataFrame or np.ndarray): Features dataset.
    y (pd.Series or np.ndarray): Target dataset.
    models (dict): A dictionary where keys are model names and values are the model instances.

    Returns:
    dict: A dictionary with model names as keys and their corresponding R2 score on the test set as values.
    """
    try:
        
        report = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[name] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e, sys)

def save_object(file_path, obj):
    """
    Saves a Python object to a file using dill.

    Parameters:
    file_path (str): The path to the file where the object will be saved.
    obj (object): The Python object to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
