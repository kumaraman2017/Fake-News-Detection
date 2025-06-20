import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, text_list):
        """
        Takes a list of news text strings and returns predicted labels.
        """
        try:
            model_path = r'artifacts/model.pkl'
            vectorizer_path = r'artifacts/preprocessor.pkl'

            # Load model and vectorizer (preprocessor)
            model = load_object(file_path=model_path)
            vectorizer = load_object(file_path=vectorizer_path)

            # Transform input text
            text_transformed = vectorizer.transform(text_list)

            # Predict
            preds = model.predict(text_transformed)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, text: str):
        self.text = text

    def get_data_as_dataframe(self):
        """
        Wraps the input text as a DataFrame to match expected input.
        """
        try:
            data_dict = {"text": [self.text]}
            return pd.DataFrame(data_dict)
        except Exception as e:
            raise CustomException(e, sys)
