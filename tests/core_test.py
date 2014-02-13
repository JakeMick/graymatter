from sklearn.datasets import make_classification
from ..graymatter import MLP


def check_if_fit():
    mlp = core.mlp()
    mlp = core.MLP()
    X, y = make_classification()
    mlp.fit(X, y)

if __name__ == "__main__":
    check_if_fit()
