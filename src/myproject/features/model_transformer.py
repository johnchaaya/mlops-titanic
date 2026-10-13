from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
)


class TitanicModelTransformer:

    def __init__(self):

        num_cat_transformation = ColumnTransformer(
            [
                (
                    "scaling",
                    MinMaxScaler(),
                    ["Age", "Fare"],
                ),
                (
                    "onehotencoding1",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["Embarked", "Pclass"],
                ),
                (
                    "ordinal",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    ["Sex"],
                ),
                (
                    "onehotencoding2",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["Title", "Family_size"],
                ),
            ],
            remainder="passthrough",
        )

        bins = ColumnTransformer(
            [
                (
                    "Kbins",
                    KBinsDiscretizer(
                        n_bins=15,
                        encode="ordinal",
                        strategy="quantile",
                    ),
                    [0, 2],
                ),
            ],
            remainder="passthrough",
        )

        self.transformer = Pipeline(
            [
                ("num_cat_transformation", num_cat_transformation),
                ("bins", bins),
            ]
        )

    def fit_transform(self, X):
        return self.transformer.fit_transform(X)

    def transform(self, X):
        return self.transformer.transform(X)