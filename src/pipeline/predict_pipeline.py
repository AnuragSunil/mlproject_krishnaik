
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.model = self.load_model()
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.preprocessor = self.load_preprocessor()

    def load_model(self):
        try:
            model = load_object(file_path=self.model_path)
            return model
        except Exception as e:
            raise CustomException(f"Failed to load model: {str(e)}", sys)

    def load_preprocessor(self):
        try:
            preprocessor = load_object(file_path=self.preprocessor_path)
            return preprocessor
        except Exception as e:
            raise CustomException(f"Failed to load preprocessor: {str(e)}", sys)

    def preprocess_data(self, features):
        try:
            # Assuming features is a DataFrame with columns matching the input format
            if not isinstance(features, pd.DataFrame):
                raise CustomException("Input features should be a Pandas DataFrame.")

            # Apply the preprocessor loaded from artifacts/preprocessor.pkl
            data_scaled = self.preprocessor.transform(features)

            return data_scaled
        
        except Exception as e:
            raise CustomException(f"Failed to preprocess data: {str(e)}", sys)

    def predict(self, features):
        try:
            if self.model is None or self.preprocessor is None:
                raise CustomException("Model or preprocessor not loaded correctly.")

            preprocessed_data = self.preprocess_data(features)
            preds = self.model.predict(preprocessed_data)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

