from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for machine learning models."""

    @abstractmethod
    def fit(self, X, y):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass