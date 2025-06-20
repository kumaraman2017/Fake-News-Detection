import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.loggers import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("🚀 Starting training for Fake News Detection with RandomizedSearchCV...")

            # ✅ Use only sparse-friendly models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000, solver="saga"),
                "Naive Bayes": MultinomialNB(),
                "Linear SVM": LinearSVC(max_iter=1000, dual=False),
            }

            # ✅ Hyperparameter distributions
            params = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10],
                    "penalty": ["l1", "l2"]
                },
                "Naive Bayes": {
                    "alpha": [0.1, 0.5, 1.0]
                },
                "Linear SVM": {
                    "C": [0.01, 0.1, 1, 10]
                },
            }

            assert set(models.keys()) == set(params.keys()), "Models and Params mismatch!"

            # ✅ RandomizedSearchCV loop
            best_model_name = None
            best_model = None
            best_score = 0.0

            for name, model in models.items():
                logging.info(f"🔍 Tuning: {name} with RandomizedSearchCV...")

                rs = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=params[name],
                    n_iter=2,   # Keep low for memory safety; increase for final runs
                    cv=2,
                    n_jobs=-1,
                    verbose=1,
                    random_state=42
                )

                rs.fit(X_train, y_train)
                y_pred = rs.predict(X_test)
                score = accuracy_score(y_test, y_pred)

                logging.info(f"✅ {name} Test Accuracy: {score:.4f} | Best Params: {rs.best_params_}")

                if score > best_score:
                    best_model_name = name
                    best_model = rs.best_estimator_
                    best_score = score

            if best_score < 0.7:
                raise CustomException(f"No satisfactory model found. Best score: {best_score:.4f}")

            logging.info(f"🏆 Best Model: {best_model_name} | Final Test Accuracy: {best_score:.4f}")

            # ✅ Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"📦 Best model saved at: {self.model_trainer_config.trained_model_file_path}")

            return best_score

        except Exception as e:
            raise CustomException(e, sys)
