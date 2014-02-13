from sklearn.datasets import make_classification
from graymatter.core import MLP


def check_if_fit():
    model = MLP()
    X, y = make_classification()
    model.fit(X, y)

if __name__ == "__main__":
    check_if_fit()
