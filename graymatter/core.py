"""
MLP estimators wrapping Pylearn2
"""

import numpy as np
from warnings import warn

from sklearn.base import BaseEstimator
from sklearn.utils import check_arrays, unique, type_of_input

from pylearn2.datasets import DenseDesignMatrix
from pylearn2.models import mlp

__all__ = ['MLP']


class MLP(BaseEstimator):

    """Multilayer perceptron trained by SGD in Pylearn2

    Parameters
    ----------
    layers : list of integers
        The hidden layer sizes. The input layer is taken to be X.shape[1]. If
        y is multioutput, the output layer is taken to be y.shape[1]. If y is
        multiclass, the output layer is taken to be unique(y).
        Defaults to [10]

    learning_rate : float, optional
        The learning rate for weight updates.
        Defaults to 0.1

    scaling_learning_rate : float, optional
        

    batch_size : int, optional
        Number of examples per minibatch.

    max_iter : int, optional
        The maximum number of epochs.

    irange_init : float, optional
        Weight initialization range for the intercepts.
        Defaults to TODO
    
    init_bias : float, optional
        Weight initilization for the biases.
        Defaults to TODO

    momentum : float, optional
        The initial training momentum.

    convolutional_input : bool, optional
        The input layer shares weights.

    hidden_layer_type : string, optional
        The type of the hidden layer. Available options are 'rect_linear',
        'sigm', ... defaults to 'rect_linear'.

    dropout : bool, optional
        Whether to use dropout or not. Defaults to False.

    dropout_probs : list of floats, 
        The dropout probability. Only used when dropout=True.
        The values are between 0.01 and 1.
        Defaults to [.8, .5, 1.0].

    verbose : int, optional.
        The verbosity level. A higher level will print convergence criteria.
        Defaults to 0.

    random_state : int or numpy.RandomState, optional
        A random number generator instance to define the state of the random
        permutations generator. If an integer is given, it fixes the seed.
        Defaults to the global numpy random number generator.

    Attributes
    ----------
    `layers_` : allocated MLP weights

    `type_of_target_` : Type of target variable, i.e. continuous,
                        continuous-multioutput, binary, multiclass,
                        mutliclass-multioutput
                        See: MLP.fit
    
    `input_size_` : How many inputs the network expects.

    `output_size_` : How many outputs the network produces.

    Examples
    --------

    >>> import numpy as np...
    """

    def __init__(self, layers=[10], learning_rate=0.1, batch_size=10,
                 max_iter=10, irange_init=0.1, init_bias=1.0, momentum=0.5,
                 convolutional_input=False, hidden_layer_type='rect_linear',
                 dropout=False, dropout_probs=[.8, .5, 1], verbose=0,
                 random_state=None):
        self.layers = layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.convolutional_input = convolutional_input
        self.hidden_layer_type = hidden_layer_type
        self.dropout = dropout
        self.dropout_probs = dropout_probs
        self.verbose = verbose
        self.random_state = random_state
        self.validate_params()

    def fit(self, X, y):
        """Fit MLP Classifier according to X, y

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples
        and n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_classes]
        Target values. It determines the problem type.

        *binary*
        If y is a vector of integers with two unique values.

        *multiclass*
        If y is a vector of integers with three or more values
        or if y is a two-dimensional array of integers and there exists only
        one non-zero element per row.

        *multiclass-multioutput*
        If y is two-dimensional array of integers with two unique values
        and there exists more than one non-zero element per row.

        *continuous*
        If y is a vector of floats.

        *continuous-multioutput*
        If y is a two-dimensional array of floats.

        Returns
        -------
        self : object
        Returns self.
        """
        X, = check_arrays(X, sparse_format='dense')

        n_samples, self.input_size_ = X.shape

        y = np.atleast_1d(y)

        self.type_of_target_ = type_of_target(y)
        if verbose > 0:
            print("The inferred type of y is %s" % self.type_of_target_)

        self.check_type_implemented()

        self._inst_mlp(n_features)
        self._fit_mlp(X, y)

    def validate_params(self):
        pass

    def check_type_implemented(self):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def _inst_mlp(self, n_features):
        # A list to hold all of the hidden and output layers.
        layers = []

        # Add the hidden layer.
        for ind in xrange(len(self.layers)):
            layer_name = 'h%i' % ind
            hidden_size = self.layers[ind]
            if self.hidden_layer_type == 'rect_linear':
                layers.append(
                    mlp.RectifiedLinear(layer_name=layer_name, hidden_size,
                                        irange=self.irange_init, init_bias))
            elif self.hidden_layer_type == 'sigmoid':
                layers.append(
                    mlp.Sigmoid(layer_name=layer_name, hidden_size,
                                irange=self.irange_init, init_bias))
            else:
                # TODO
                # Not implemented error?
                raise('derp')

        # Add the output layer.
        if self.type_of_target in ['binary', 'multiclass-multioutput']:
            layers.append(mlp.Sigmoid(layer_name='output',
                                      self.output_size_,
                                      irange=self.irange_init))
        elif self.type_of_target == 'multiclass':
            layers.append(mlp.SoftMax(layer_name='output',
                                      self.output_size_
                                      irange=self.irange_init))
        else:
            # TODO
            # Not implemented error?
            raise('derp')
        # Create the ANN object for pylearn2
        self.network_ = mlp.MLP(layers, nvis=self.n_features)
