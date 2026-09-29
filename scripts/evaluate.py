import argparse
import json
import pickle
import os

import pandas as pd
from sklearn.metrics import accuracy_score


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate Titanic model"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=False)

    return parser


def main():
    args = build_parser().parse_args()

    # Read evaluation data
    df = pd.read_csv(args.input)

    # Separate features and target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Load trained model
    with open(args.model, "rb") as file:
        model = pickle.load(file)

    # Make predictions
    predictions = model.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y, predictions)

    print("Evaluation completed")
    print("Accuracy:", accuracy)

    # Optional: save metric to JSON
    if args.output:
        output_dir = os.path.dirname(args.output)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        metrics = {
            "accuracy": accuracy
        }

        with open(args.output, "w") as file:
            json.dump(metrics, file, indent=4)

        print("Metrics saved to:", args.output)


if __name__ == "__main__":
    main()