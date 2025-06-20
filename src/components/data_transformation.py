import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.loggers import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Create and return a TF-IDF Vectorizer for text data.
        """
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_df=0.7
            )
            logging.info("Created TF-IDF Vectorizer for text preprocessing")
            return vectorizer

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Loads train/test CSVs, fits TF-IDF on train, transforms both,
        saves the vectorizer, and returns arrays ready for training.
        """
        try:
            # ✅ Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Loaded train and test datasets")

            # ✅ Use 'title' + 'text' if needed. Here, using 'title' only:
            X_train = train_df['title']
            y_train = train_df['label']
            X_test = test_df['title']
            y_test = test_df['label']

            # ✅ Get TF-IDF Vectorizer
            preprocessor = self.get_data_transformer_object()

            # ✅ Fit on train, transform train + test
            logging.info("Fitting TF-IDF Vectorizer on training data")
            X_train_transformed = preprocessor.fit_transform(X_train)
            logging.info("Transforming test data with TF-IDF Vectorizer")
            X_test_transformed = preprocessor.transform(X_test)

            # ✅ Save the vectorizer
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            logging.info("TF-IDF Vectorizer saved")

            # ✅ Return sparse matrices as arrays + labels
            return (
                X_train_transformed,  # sparse matrix
                X_test_transformed,   # sparse matrix
                y_train.values,
                y_test.values,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
