from sklearn.linear_model import LogisticRegressionCV

from myproject.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):

    def __init__(self):
        self.model = LogisticRegressionCV()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)