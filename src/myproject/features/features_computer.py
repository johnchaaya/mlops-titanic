import pandas as pd

from myproject.features.base_features_computer import BaseFeaturesComputer


class TitanicFeaturesComputer(BaseFeaturesComputer):

    def family_size(self, number):
        if number == 1:
            return "Alone"
        elif number > 1 and number < 5:
            return "Small"
        else:
            return "Large"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Extract title
        df["Title"] = (
            df["Name"]
            .str.split(", ", expand=True)[1]
            .str.split(".", expand=True)[0]
        )

        # Group rare titles
        df["Title"] = df["Title"].replace(
            [
                "Lady",
                "the Countess",
                "Capt",
                "Col",
                "Don",
                "Dr",
                "Major",
                "Rev",
                "Sir",
                "Jonkheer",
                "Dona",
            ],
            "Rare",
        )

        df["Title"] = df["Title"].replace("Mlle", "Miss")
        df["Title"] = df["Title"].replace("Ms", "Miss")
        df["Title"] = df["Title"].replace("Mme", "Mrs")

        # Create family size
        df["Family_size"] = df["SibSp"] + df["Parch"] + 1
        df["Family_size"] = df["Family_size"].apply(self.family_size)

        # Drop unused columns
        df.drop(
            columns=["Name", "Parch", "SibSp", "Ticket"],
            inplace=True,
        )

        df["Age"] = df["Age"].astype("int64")

        df.drop(columns=["PassengerId"], inplace=True)

        return df