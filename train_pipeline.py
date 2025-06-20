import sys
import os
import logging

import numpy as np
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

# ✅ Proper logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if __name__ == "__main__":
    logging.info("✅ Training pipeline started.")
    try:
        # 1️⃣ Data Ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info(f"📁 Data ingestion complete: Train -> {train_path}, Test -> {test_path}")

        # 2️⃣ Data Transformation
        transformation = DataTransformation()
        X_train_transformed, X_test_transformed, y_train, y_test, vectorizer_path = (
            transformation.initiate_data_transformation(train_path, test_path)
        )
        logging.info(f"🔄 Data transformation complete. Vectorizer saved at: {vectorizer_path}")

        logging.info(f"🧩 X_train shape: {X_train_transformed.shape} | X_test shape: {X_test_transformed.shape}")

        # ✅ 3️⃣ Model Training — pass sparse X & y directly
        trainer = ModelTrainer()
        trainer.initiate_model_trainer(
            X_train_transformed,  # sparse!
            X_test_transformed,   # sparse!
            y_train,              # np.array
            y_test                # np.array
        )
        logging.info("🏆 Model training completed successfully.")

    except Exception as e:
        logging.error(f"❌ Exception occurred: {e}")
        raise CustomException(e, sys)
