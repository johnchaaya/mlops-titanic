import argparse
import pickle
import os

import pandas as pd


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

    # Read feature data
    df = pd.read_csv(args.input)

    # Load trained model
    with open(args.model, "rb") as file:
        model = pickle.load(file)

    # Make predictions
    predictions = model.predict(df)

    # Create predictions dataframe
    output_df = pd.DataFrame({
        "Survived": predictions
    })

    # Create output folder if needed
    output_dir = os.path.dirname(args.output)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save predictions
    output_df.to_csv(args.output, index=False)

    print("Prediction completed")
    print("Number of predictions:", len(predictions))
    print("Predictions saved to:", args.output)


if __name__ == "__main__":
    main()