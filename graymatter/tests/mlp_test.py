from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import log_loss
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from ..mlp import MLP


class TestMLP(object):

    def __init__(self):
        self.X_cl, self.y_cl = make_classification(100)
        self.X_mcl, self.y_mcl = make_classification(
            1000, n_informative=5, n_classes=3)
        self.X_re, self.y_re = make_regression(100)

    def test_if_fit_classification(self):
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
        model = MLP(max_iter=100, dropout='True')
        model.fit(self.X_re, self.y_re)
        linear = LinearRegression()
        linear.fit(model.predict(self.X_re).reshape(-1, 1),
                   self.y_re.reshape(-1, 1))
        assert(model.type_of_target_ == 'continuous')
        assert (np.abs(1 - linear.coef_[0]) < 0.05)
        assert (np.abs(linear.intercept_) < 0.05)

    def test_accuracy(self):
        model = MLP()
        model.fit(self.X_cl[:50], self.y_cl[:50])
        y_pred = model.predict_proba(self.X_cl[50:])
        ll = log_loss(self.y_cl[50:], y_pred)
        assert(ll < .05)

    def test_multiclass(self):
        model = MLP()
        model.fit(self.X_mcl, self.y_mcl)
