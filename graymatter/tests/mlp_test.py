from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import log_loss
from ..mlp import MLP


def test_if_fit_classification():
    model = MLP()
    X, y = make_classification()
    model.fit(X, y)


def test_if_fit_regression():
    model = MLP()
    X, y = make_regression()
    model.fit(X, y)


def test_sigmoid():
    model = MLP(hidden_layer_type='sigm')
    X, y = make_classification()
    model.fit(X, y)


def test_dropout():
    model = MLP(dropout='True')
    X, y = make_classification()
    model.fit(X, y)


def test_accuracy():
    model = MLP()
    X, y = make_classification(100)
    model.fit(X[:50], y[:50])
    y_pred = model.predict_proba(X[50:])
    ll = log_loss(y[50:], y_pred)
    assert(ll < .05)

if __name__ == "__main__":
    check_if_fit()
