import pandas as pd

from myproject.preprocessing.base_preprocessor import BasePreprocessor


class TitanicPreprocessor(BasePreprocessor):

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Drop Cabin
        df.drop(columns=["Cabin"], inplace=True)

        # Fill missing Embarked
        df["Embarked"] = df["Embarked"].fillna("S")

        # Fill missing Fare
        df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

        # Fill missing Age
        df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
            lambda x: x.fillna(x.median())
        )

        return df