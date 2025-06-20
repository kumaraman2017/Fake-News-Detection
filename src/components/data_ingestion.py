import os
import sys
from src.exception import CustomException
from src.loggers import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "merged_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component for Fake News dataset")

        try:
            # ✅ Updated paths: data/True.csv and data/Fake.csv (NOT inside notebook/)
            true_df = pd.read_csv(os.path.join('data', 'True.csv'))
            fake_df = pd.read_csv(os.path.join('data', 'Fake.csv'))
            logging.info("Loaded True.csv and Fake.csv from data folder")

            # Add labels
            true_df['label'] = 1  # True news
            fake_df['label'] = 0  # Fake news

            # Combine datasets
            df = pd.concat([true_df, fake_df], ignore_index=True)
            logging.info(f"Merged dataset shape: {df.shape}")

            # Save raw merged data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # Split train/test
            train_set, test_set = train_test_split(
                df, 
                test_size=0.2, 
                random_state=42, 
                stratify=df['label']
            )
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Ingestion of Fake News data completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
