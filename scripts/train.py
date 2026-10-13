import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from myproject.features.model_transformer import TitanicModelTransformer
from myproject.models.logistic_regression_model import LogisticRegressionModel
from myproject.models.model_pipeline import ModelPipeline

from myproject.models.random_forest_model import RandomForestModel

def build_parser():
    parser = argparse.ArgumentParser(
        description="Train Titanic survival model"
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--eval-output", required=True)
    parser.add_argument(
    "--model-type",
    choices=["logistic", "random_forest"],
    default="logistic",
)

    return parser


def main():
    args = build_parser().parse_args()

    df = pd.read_csv(args.input)

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_eval, y_train, y_eval = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    transformer = TitanicModelTransformer()
    if args.model_type == "logistic":
        model = LogisticRegressionModel()

    elif args.model_type == "random_forest":
        model = RandomForestModel()

    pipeline = ModelPipeline(
        transformer=transformer,
        model=model,
    )

    pipeline.fit(X_train, y_train)

    pipeline.save(args.output)

    eval_df = X_eval.copy()
    eval_df["Survived"] = y_eval.values
    eval_df.to_csv(args.eval_output, index=False)

    print("Training completed")
    print("Training rows:", len(X_train))
    print("Evaluation rows:", len(X_eval))
    print("Model type:", args.model_type)
    print("Model saved to:", args.output)


if __name__ == "__main__":
    main()