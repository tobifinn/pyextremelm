# -*- coding: utf-8 -*-
"""
Created on 19.05.16
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
from copy import deepcopy
from time import time

# External modules
import numpy as np

# Internal modules
from .activations import named_activations, unnamed_activations

__version__ = "0.1"


class ELMBase(object):
    """
    The base class for every elm component in this layered approach.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, X, y=None):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def update(self, X, y=None):
        pass


class ExtremeLearningMachine(ELMBase):
    """
    This class is a container for the layers of the extreme learning machine.

    Attributes:
        layers (List[ELMLayer]):
            The list with the added layers for the extreme learning machine.
    """

    def __init__(self):
        self.layers = []
        # self.loss = ELMMetric(loss)

    def add_layer(self, layer):
        """
        This method adds a pre-configured layer to the layer list.
        Args:
            layer (Instance of child of ELMLayer): The pre-configured layer.

        Returns:
            self: returns an instance of self.
        """
        self.layers.append(layer)
        return self

    def fit(self, X, y=None):
        """
        This method trains the elm for supervised and unsupervised training.
        Args:
            X (numpy array): The input data field.
            y (optional[numpy array]): The output data field.
                In unsupervised learning this is only for consistency.

        Returns:
            trainings_info (dict): A dict with trainings informations.
                samples (integer): How many samples are used to train.
                loss (float): An error approximation for the training dataset,
                    for the given loss function of the network.
                train_time (float): The training duration for the whole
                    network in seconds.
                training_output (numpy array): The forecasted output for the
                    training dataset. Could be used for further network steps.

        """
        training_info = {"samples": None, "loss": None, "train_time": None,
                         "output": None}
        t0 = time()
        training_info["samples"] = X.shape[0]
        output = X
        for layer in self.layers:
            input = output
            output = layer.fit(input, y)
        # self.loss(y, loss)
        training_info['output'] = output
        training_info['train_time'] = time() - t0
        return training_info

    def predict(self, X):
        """
        This method predict the outcome for given input x.
        Args:
            X (numpy array): The input data field.

        Returns:
            prediction (numpy array): The predicted data field.
            prediction_time (float): The prediction duration in seconds.
        """
        t0 = time()
        prediction = X
        for layer in self.layers:
            input = prediction
            prediction = layer.predict(input)
        return prediction

    def update(self, X, y=None):
        pass

    def print_network_structure(self):
        s = [str(l) for l in self.layers]
        s = '\n'.join(s)
        return '\033[1m' + 'Network structure\n' + '\033[0m' + s


class ELMLayer(ELMBase):
    """
    The ELMLayer represents one layer within the extreme learning machine,

    Attributes:
        n_neurons (int): Number of neurons within the layer.
        train_algorithm (Child of ELMTraining): Training method of the layer.
        activation_funct (str or numpy function):
            The function with which the values should be activated,
            Default is None, because in some layers there is no activation.
    """

    def __init__(self, n_neurons, activation="linear",
                 bias=False):
        """
        Args:
            n_neurons (int): Number of neurons within the layer.
            activation_funct (optional[str or activation function]):
                The function with which the values should be activated,
                Default is a linear activation.
            bias (bool): If the layer should have a bias. Default is False.
        """
        self.n_neurons = n_neurons
        self.weights = {"input": None, "bias": None}
        self.activation_funct = self.get_activation(activation)
        self.bias = bias

    def __str__(self):
        s = "{0:s}(neurons: {1:d}, activation: {2:s}, bias: {3:s})".format(
            self.__class__.__name__, self.n_neurons,
            str(type(self.activation_funct).__name__), str(self.bias))
        return s

    @abc.abstractmethod
    def train_algorithm(self, X, y):
        pass

    def fit(self, X, y=None):
        self.weights = self.train_algorithm(X, y)
        X = self.add_bias(X)
        self.activation_funct = self.activation_funct(self.weights)
        return self.activation_funct.activate(X)

    def predict(self, X, **kwargs):
        X = self.add_bias(X)
        return self.activation_funct.activate(X)

    def update(self, X, y=None, decay=1):
        pass

    def add_bias(self, X):
        if self.bias:
            input_dict = {"input": X, "bias": np.ones(X.shape[0])}
        else:
            input_dict = {"input": X, "bias": None}
        return input_dict

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
