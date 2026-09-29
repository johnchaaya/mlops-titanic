import argparse
import pickle
import os

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train Titanic survival model"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)

    return parser


def main():
    args = build_parser().parse_args()

    # Read feature data
    df = pd.read_csv(args.input)

    # Separate features and target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Split data
    X_train, X_eval, y_train, y_eval = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Column types
    numeric_features = ["Pclass", "Age", "Fare"]

    categorical_features = [
        "Sex",
        "Embarked",
        "Title",
        "Family_size",
    ]

    # Transformations
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    # Model pipeline
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    # Train model
    model.fit(X_train, y_train)

    # Create model folder if necessary
    output_dir = os.path.dirname(args.output)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save model
    with open(args.output, "wb") as file:
        pickle.dump(model, file)

    # Save evaluation data
    eval_df = X_eval.copy()
    eval_df["Survived"] = y_eval.values
    eval_df.to_csv(args.eval_output, index=False)

    print("Training completed")
    print("Training rows:", len(X_train))
    print("Evaluation rows:", len(X_eval))
    print("Model saved to:", args.output)


if __name__ == "__main__":
    main()