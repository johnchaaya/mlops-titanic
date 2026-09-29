import argparse
import pandas as pd


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create features for Titanic data"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    return parser


def family_size(number):
    if number == 1:
        return "Alone"
    elif number > 1 and number < 5:
        return "Small"
    else:
        return "Large"


def main():
    args = build_parser().parse_args()

    # Read preprocessed data
    df = pd.read_csv(args.input)

    # Extract title from passenger name
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

    # Convert family size into categories
    df["Family_size"] = df["Family_size"].apply(family_size)

    # Remove columns not used by the model
    df.drop(
        columns=["Name", "Parch", "SibSp", "Ticket"],
        inplace=True,
    )

    # Convert Age to integer
    df["Age"] = df["Age"].astype("int64")

    # PassengerId is not used for training
    df.drop(columns=["PassengerId"], inplace=True)

    # Save features
    df.to_csv(args.output, index=False)

    print("Feature engineering completed")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()