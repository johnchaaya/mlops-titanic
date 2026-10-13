import os
import pickle


class ModelPipeline:

    def __init__(self, transformer, model):
        self.transformer = transformer
        self.model = model

    def fit(self, X, y):
        X_transformed = self.transformer.fit_transform(X)
        self.model.fit(X_transformed, y)

    def predict(self, X):
        X_transformed = self.transformer.transform(X)
        return self.model.predict(X_transformed)

    def save(self, path):
        output_dir = os.path.dirname(path)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)