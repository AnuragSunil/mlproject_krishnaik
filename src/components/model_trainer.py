import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                    'splitter': ['best', 'random'],
                },
                "Random Forest Regressor": {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.5],
                    'max_depth': [3, 5, 7],
                },
                "Linear Regression": {},
                "Lasso": {},
                "K-Neighbors Regressor": {},
                "XGBRegressor": {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.5],
                    'max_depth': [3, 5, 7],
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.5],
                },
            }

            model_report = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            best_model.fit(X_train, y_train)  # Ensure the best model is fitted

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
