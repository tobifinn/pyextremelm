# -*- coding: utf-8 -*-
"""
Created on 02.08.16

Created for pyextremelm

@author: Tobias Sebastian Finn, tobias.sebastian.finn@studium.uni-hamburg.de

    Copyright (C) {2016}  {Tobias Sebastian Finn}

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
# System modules
import abc

# External modules
import numpy as np

# Internal modules
from ..activations import dense as dense_act, convolutional as conv_act

__version__ = "0.1"


class ELMLayer(object):
    """
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_features, activation='linear', bias=True):
        """
        The ELMLayer represents one layer within the neural network.
        Args:
            n_features (int): How many features/neurons should the layer have.
            activation (str/dense activation function):
                Which activation should be used to activate the layer.
            bias (bool): If the layer should contain an intercept.
        """
        self.n_features = n_features
        self.bias = bias
        self.weights = {"input": None, "bias": None}
        self.activation_fct = self.get_activation(activation)
        if type(self.activation_fct).__name__ == 'type':
            self.activation_fct = self.activation_fct()

    def __str__(self):
        s = "{0:s}(neurons: {1:d}, activation: {2:s}, bias: {3:s})".format(
            self.__class__.__name__, self.n_features,
            str(type(self.activation_fct).__name__), str(self.bias))
        return s

    @abc.abstractmethod
    def fit(self, X, y=None):
        """
        Method to fit the layer.
        Args:
            X (numpy array): The input data array.
            y (optional[numpy array]): The target data array.
                For unsupervised layers, like random or autoencoder layers,
                the target isn't needed. Default is None.

        Returns:
            fitted_X (numpy array): The output data array of the layer.
                It will have the form: g(x,A,b), with g the activation
                function, A the input weights and b the bias weights.
        """
        pass

    def predict(self, X):
        """
        Method to make a prediction with the layer.
        Args:
            X (numpy array): The input data array for the layer.

        Returns:
            prediction (numpy array): The predicted data array of the layer.
                It will have the form: g(x,A,b), with g the activation
                function, A the input weights and b the bias weights.
        """
        X = self.add_bias(X)
        return self.activation_fct.activate(X)

    @abc.abstractmethod
    def update(self, X, y=None, decay=1):
        """
        Method to update the layer weights with given arrays.
        Args:
            X (numpy array): The input data array.
            y (optional[numpy array]): The target data array.
                For unsupervised layers, like random or autoencoder layers,
                the target isn't needed. Default is None.
            decay (optional[float]): The decay factor determines the decaying
                of old partial weights. The update is calculated by this
                formula: beta_new = beta_old + decay*delta_beta_new.
                This update formula with a decay factor of 1 was proposed by
                McDonnell, M. D. et al. (2015) [1]. Default is 1.

        Returns:
            fitted_X (numpy array): The output data array of the layer.
                It will have the form: g(x,A,b), with g the activation
                function, A the input weights and b the bias weights.
        -----------------------------------------------------------------------
        [1] McDonnell, M. D., Tissera, M. D. et al. (2015).
            Fast, simple and accurate handwritten digit classification by
            training shallow neural network classifiers with the
            ‘extreme learning machine’algorithm. PloS one, 10(8)
        """
        pass

    def add_bias(self, X):
        """
        Method to add an intercept column t othe X if the layer should have a
        bias.
        Args:
            X (numpy array): Array to which the intercept column should be
                added. Input dim is rows x cols.

        Returns:

        """
        if self.bias:
            input_dict = {"input": X, "bias": np.ones(X.shape[0])}
        else:
            input_dict = {"input": X, "bias": None}
        return input_dict

    @staticmethod
    def get_activation(funct="sigmoid"):
        """
        Function to get the activation function
        Args:
            funct (str or function): The function name or the function itself.

        Returns:
            function: The activation function.
        """
        if isinstance(funct, str) and funct in dense_act.named_activations:
            return dense_act.named_activations[funct]
        elif funct is None:
            return dense_act.named_activations["linear"]
        else:
            return funct

    @staticmethod
    def get_dim(X):
        """
        Get the dimensions of X.
        Args:
            X (numpy array): X is the input array (shape: samples*dimensions).

        Returns:
            dimensions (int): The dimensions of X.
        """
        return X.shape[1] if len(X.shape) > 1 else 1


class ELMConvLayer(ELMLayer):
    def __init__(self, n_features, spatial=(3,3), stride=(1,1), pad=(1,1),
                 activation='linear', bias=True):
        super().__init__(n_features, activation, bias)
        self.activation_fct = self.get_activation(activation)
        self.spatial = spatial
        self.stride = stride
        self.pad = pad
        self.conv_fct = None
        self.filter_shape = None

    def __str__(self):
        s = "{0:s}, spatial extent: {1:s}, stride: {2:s}, padding: {3:s})".\
            format(super().__str__()[:-1], str(self.spatial), str(self.stride),
                   str(self.pad))
        return s

    @staticmethod
    def get_activation(funct="sigmoid"):
        """
        Function to get the activation function
        Args:
            funct (str or function): The function name or the function itself.

        Returns:
            function: The activation function
        """
        if isinstance(funct, str) and funct in conv_act.named_activations:
            return conv_act.named_activations[funct]
        else:
            return funct

    @abc.abstractmethod
    def _build_conv(self):
        """
        Method to build the convolutional function.
        """
        pass

    @abc.abstractmethod
    def fit(self, X, y=None):
        pass

    @abc.abstractmethod
    def update(self, X, y=None, decay=1):
        pass

    def predict(self, X):
        """
        Method to make a prediction with the layer.
        Args:
            X (numpy array): The input data array for the layer.

        Returns:
            prediction (numpy array): The predicted data array of the layer.
                It will have the form: g(x,A,b), with g the activation
                function, A the input weights and b the bias weights.
        """
        return self.conv_fct(X)

