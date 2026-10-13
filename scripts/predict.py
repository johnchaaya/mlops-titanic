import argparse
import os

import pandas as pd

from myproject.models.base_model import BaseModel


def build_parser():
    parser = argparse.ArgumentParser(
        description="Predict Titanic survival"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)

    return parser


def main():
    args = build_parser().parse_args()

    df = pd.read_csv(args.input)

    model = BaseModel.load(args.model)

    predictions = model.predict(df)

    output_df = pd.DataFrame(
        {
            "Survived": predictions
        }
    )

    output_dir = os.path.dirname(args.output)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_df.to_csv(args.output, index=False)

    print("Prediction completed")
    print("Number of predictions:", len(predictions))
    print("Predictions saved to:", args.output)


if __name__ == "__main__":
    main()