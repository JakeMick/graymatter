[![Build Status](https://travis-ci.org/JakeMick/graymatter.png?branch=master)](https://travis-ci.org/JakeMick/graymatter)

Graymatter
==========
Graymatter provides an adaptor on top of the [pylearn2](deeplearning.net/software/pylearn2/)
GPU neural network library in the style of [scikit-learn](http://scikit-learn.org/stable/)
for integration with scikit-learn's utilities. Graymatter is not affliated with
either project. (Though I'll provide upstream bug-fixes when they arise.)

Currently there's a testsuite for parameter configurations. Some parameter
combinations result in strange behavior in the compiled theano that throws
errors in pylearn2, such as a high learning rate with a high max column norm.
I'm working on catching them at init time.

The purpose of this project is to create a stable api over some of the core
functionality of pylearn2.models.mlp. Pull requests and github issues for
bug-fixes and feature-enhancement are welcome.

The code is mostly self-documenting in the docstrings. In IPython,
graymatter.MLP? has a complete list of the configurable parameters.

This is licensed under MIT, so you can fork/merge/whatever you want with it.

Requirements
============

    pylearn2
    scikit-learn

Using the GPU
=============
Create a file, ~/.theanorc with the following contents.

    [global]
    floatX = float32
    device = gpu0
    
    [nvcc]
    fastmath = True

When you initialize the model, STDOUT should be something like

    Using gpu device 0: GeForce GTX 560 Ti


Example
=======
    from graymatter.nn import MLP
    from sklearn.datasets import make_classification

    X, y = make_classification(100)
    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]

    model = MLP(dropout=True)
    model.fit(X_train, y_train)
    model.predict(X_test)

