import argparse
import pandas as pd


def build_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess Titanic data"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    return parser


def main():
    args = build_parser().parse_args()

    # Read data
    df = pd.read_csv(args.input)

    # Drop Cabin
    df.drop(columns=["Cabin"], inplace=True)

    # Fill missing Embarked
    df["Embarked"] = df["Embarked"].fillna("S")

    # Fill missing Fare
    df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

    # Fill missing Age using median by Sex and Pclass
    df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    # Save cleaned data
    df.to_csv(args.output, index=False)

    print("Preprocessing completed")
    print("Shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()