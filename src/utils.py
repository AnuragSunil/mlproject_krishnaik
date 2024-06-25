import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluates multiple regression models on the given dataset and returns a report
    with the R2 score for the test set of each model.

    Parameters:
    X_train (np.ndarray): Training features dataset.
    y_train (np.ndarray): Training target dataset.
    X_test (np.ndarray): Test features dataset.
    y_test (np.ndarray): Test target dataset.
    models (dict): A dictionary where keys are model names and values are the model instances.
    params (dict): A dictionary where keys are model names and values are the parameter grids for GridSearchCV.

    Returns:
    dict: A dictionary with model names as keys and their corresponding R2 score on the test set as values.
    """
    try:
        report = {}

        for model_name, model in models.items():
            if model_name in params:
                param_grid = params[model_name]
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
                gs.fit(X_train, y_train)
                model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

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

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
        
        
    except Exception as e:
        raise CustomException(e,sys)
