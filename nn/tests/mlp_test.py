from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import log_loss
from ..mlp import MLP


class TestFitting(object):

    def __init__(self):
        self.X_cl, self.y_cl = make_classification(100)
        self.X_re, self.y_re = make_classification(100)

    def test_if_fit_classification():
        model = MLP()
        model.fit(self.X_cl, self.y_cl)
        assert(model.type_of_target_ == 'binary')

    def test_if_fit_regression(self):
        model = MLP()
        model.fit(self.X_re, self.y_re)
        assert(model.type_of_target_ == 'continuous')

    def test_sigmoid(self):
        model = MLP(hidden_layer_type='sigm')
        model.fit(self.X_cl, self.y_cl)

    def test_dropout(self):
        model = MLP(dropout='True')
        model.fit(self.X_cl, self.y_cl)

    def test_accuracy(self):
        model = MLP()
        model.fit(self.X_cl[:50], self.y_cl[:50])
        y_pred = model.predict_proba(self.X_cl[50:])
        ll = log_loss(self.y_cl[50:], y_pred)
        assert(ll < .05)
