import argparse

import pandas as pd

from myproject.preprocessing.preprocessor import TitanicPreprocessor


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

    # Use preprocessing class
    preprocessor = TitanicPreprocessor()
    df = preprocessor.process(df)

    # Save cleaned data
    df.to_csv(args.output, index=False)

    print("Preprocessing completed")
    print("Shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()