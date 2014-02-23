[![Build Status](https://travis-ci.org/JakeMick/graymatter.png?branch=master)](https://travis-ci.org/JakeMick/graymatter)

Graymatter
==========
Graymatter provides an adaptor on top of the [pylearn2](deeplearning.net/software/pylearn2/)
GPU neural network library in the style of [scikit-learn](http://scikit-learn.org/stable/)
for integration with scikit-learn's utilities. Graymatter is not affliated with
either project.

Example
=======
    from graymatter import MLP
    from sklearn.datasets import make_classification

    X, y = make_classification(100)
    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]

    model = MLP(dropout=True)
    model.fit(X_train, y_train)
    model.predict(X_test)

Functionality
=============
The interface to pylearn2 is simplified. As a result, it will never
be possible to create all the functionality you could with YAML in pylearn2.

For instance,

 - Weight initilization parameters are shared across the layers.
 - Arbitrary hidden layer configurations aren't possible.

There's no simple way to create a tree-like API in the spirit of scikit-learn.

Currently,

 - Units per hidden layer are configurable.
 - Type of hidden layers are configurable.
 - Dropout is configurable.
 - Learning rate with decay is configurable.
 - Momentum with saturation is configurable.
 - Sigmoidal and Rectified Linear layers are available.
 - Weight initialization is configurable.

In the future,
 - maxout and support for convolutional inputs
 - additional neural network types
 - unsupervised pretraining
 - serialization support

The design departs from scikit-learn. Instead of providing
MLPClassifier and MLPRegressor, Graymatter provides MLP which infers the type
of y. Multiclass and multioutput regression types are handled. If this bugs you
you can pass type\_of\_y as an input argument to assert what you expect y to be.

The purpose of this project is to create a stable api over some of the core
functionality of pylearn2.models.mlp. Pull requests and github issues for
bug-fixes and feature-enhancement are welcome.

The code is self-documenting in the docstrings. In IPython,
graymatter.MLP? has a complete list of the configurable parameters.

This is licensed under MIT, so you can fork/merge/whatever you want with it.

Requirements
============

    pylearn2
    scikit-learn

Pylearn2 is install from git. They don't have a release cycle, as far as I can
tell.

Installation
============

    python setup.py build
    python setup.py install

Testing
=======
In the repository directory

    nosetests

Using the GPU
=============
This assumes you're using Ubuntu 12.04 with a working pylearn2 installation,

    sudo apt-get install libblas-dev gfortran g++ liblapack-dev

NVIDIA provides a deb package for CUDA

    wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1204/x86_64/cuda-repo-ubuntu1204_5.5-0_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu1204_5.5-0_amd64.deb
    sudo apt-get update
    sudo apt-get install cuda

Add the following lines to your ~/.bashrc

    export PATH=/usr/local/cuda-5.5/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-5.5/lib64:$LD_LIBRARY_PATH

Create a file, ~/.theanorc with the following contents.

    [global]
    floatX = float32
    device = gpu0
    
    [nvcc]
    fastmath = True

When you initialize the model, STDOUT should be something like

    Using gpu device 0: GeForce GTX 560 Ti

Scaling your inputs
===================
Neural networks are very sensitive to scaling and we're using float32. Scale
your inputs if they're not close to a mean of 0, std of 1. If you're performing
regression or multioutput regression, then it's advisable to scale those as well.
