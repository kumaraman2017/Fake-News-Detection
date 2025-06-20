import os
import sys
import dill

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

try:
    from src.exception import CustomException  # Make sure src/exception.py exists and defines CustomException
except ImportError:
    from exception import CustomException


def save_object(file_path, obj):
    """
    Saves any Python object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a Python object saved with dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Trains multiple classification models with GridSearchCV,
    evaluates accuracy, and returns a report dictionary.
    """
    try:
        report = {}

        for name, model in models.items():
            para = params[name]

            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            report[name] = {
                "best_params": gs.best_params_,
                "train_score": train_model_score,
                "test_score": test_model_score
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
if __name__ == "__main__":
    print("✅ Running utils as a module for test...")
    obj = {"foo": "bar"}
    save_object("test.pkl", obj)
    print("✅ Loaded:", load_object("test.pkl"))

