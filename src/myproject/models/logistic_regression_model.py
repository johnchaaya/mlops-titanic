import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from myproject.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):

    def __init__(self):
        numeric_features = [
            "Pclass",
            "Age",
            "Fare",
        ]

        categorical_features = [
            "Sex",
            "Embarked",
            "Title",
            "Family_size",
        ]

        preprocessing = ColumnTransformer(
            [
                (
                    "numeric",
                    StandardScaler(),
                    numeric_features,
                ),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features,
                ),
            ]
        )

        self.model = Pipeline(
            [
                ("preprocessing", preprocessing),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000),
                ),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)