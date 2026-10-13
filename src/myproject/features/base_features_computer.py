from abc import ABC, abstractmethod

import pandas as pd


class BaseFeaturesComputer(ABC):
    """Abstract base class for feature engineering."""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features and return the transformed dataframe."""
        pass