from abc import ABC, abstractmethod
import pickle
import os

import pandas as pd


class BaseModel(ABC):
    """Abstract base class for machine learning models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        """Make predictions."""
        pass

    def save(self, path: str):
        """Save the model."""
        output_dir = os.path.dirname(path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path: str):
        """Load a saved model."""
        with open(path, "rb") as file:
            return pickle.load(file)