import argparse

import pandas as pd

from myproject.features.features_computer import TitanicFeaturesComputer


def build_parser():
    parser = argparse.ArgumentParser(
        description="Create features for Titanic data"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    return parser


def main():
    args = build_parser().parse_args()

    df = pd.read_csv(args.input)

    features_computer = TitanicFeaturesComputer()
    df = features_computer.compute(df)

    df.to_csv(args.output, index=False)

    print("Feature engineering completed")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()