from sklearn.ensemble import RandomForestClassifier

from myproject.models.base_model import BaseModel


class RandomForestModel(BaseModel):

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)