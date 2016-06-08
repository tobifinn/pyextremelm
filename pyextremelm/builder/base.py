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
from .helpers import get_activation, add_bias

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


class ExtremeLearningMachine(ELMBase):
    """
    This class is a container for the layers of the extreme learning machine.

    Attributes:
        layers (List[ELMLayer]):
            The list with the added layers for the extreme learning machine.
    """
    def __init__(self, metric, iterations=1):
        self.layers = []
        self.training_info = {"trained": False, "samples": None,
                              "metric": metric, "accuracy": None,
                              "iterations": iterations, "layer_config": [],
                              "train_time": None}

    def add_existing_layer(self, layer):
        """
        This method adds a pre-configured layer to the layer list.
        Args:
            layer (Instance of child of ELMLayer): The pre-configured layer.

        Returns:
            self: returns an instance of self.
        """
        self.layers.append(layer)
        return self

    def add_new_layer(self, n_neurons, train_algorithm, activation="sigmoid"):
        """
        This method creates a new layer and adds it to the layer list.

        Args:
            n_neurons (int): Number of neurons in the layer.
            train_algorithm (ELMTraining): The training algorithm of the layer.
            activation (optional[str or numpy function]):
                The activation function for the layer.

        Returns:
            self: returns an instance of self.
        """
        layer = ELMLayer(n_neurons, train_algorithm, activation)
        self.add_existing_layer(layer)
        return self

    def fit(self, X, y=None):
        """
        This method trains the elm for supervised and unsupervised training.
        Args:
            X (numpy array): The input data field.
            y (optional[numpy array]): The output data field.
                In unsupervised learning this is only for consistency.

        Returns:
            self: returns an instance of self.
        """
        t0 = time()
        self.training_info["samples"] = X.shape[0]
        for i in range(self.training_info['iterations']):
            layers_fitted = deepcopy(self.layers)
            output = X
            for layer in layers_fitted:
                input=output
                output = layer.fit(input, y)
            if self.training_info['accuracy'] is None or \
                    self.training_info["metric"](output, y).better(
                        self.training_info['accuracy']):
                try:
                    self.training_info['accuracy'] = self.training_info["metric"](
                        y, output).score
                except:
                    pass
                self.training_info['layer_config'] = layers_fitted
        self.training_info['train_time'] = time()-t0
        self.layers = self.training_info['layer_config']
        self.training_info['trained'] = True
        return self

    def predict(self, X):
        """
        This method predict the outcome for given input x.
        Args:
            X (numpy array): The input data field.

        Returns:
            prediction (numpy array): The predicted data field.
        """
        prediction = X
        for layer in self.layers:
            input = prediction
            prediction = layer.predict(input)
        return prediction


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

    def __init__(self, n_neurons, train_algorithm, activation="linear",
                 bias=False, **kwargs):
        """
        Args:
            n_neurons (int): Number of neurons within the layer.
            train_algorithm (function from training directory):
                Training algorithm of the layer.
            activation_funct (optional[str or activation function]):
                The function with which the values should be activated,
                Default is a linear activation.
            bias (bool): If the layer should have a bias. Default is False.
        """
        self.n_neurons = n_neurons
        self.train_algorithm = train_algorithm
        self.activation_funct = get_activation(activation)
        self.weights = None
        self.bias = bias
        self.kwargs = kwargs

    @property
    def trained(self):
        """
        Property to check if the layer is trained.
        Returns:
            trained (bool): True if the layer is trained, else False.
        """
        if self.weights is None:
            trained = False
        else:
            trained = True
        return trained

    def fit(self, X, y=None):
        self.weights = self.train_algorithm(X, y, self.n_neurons, self.bias, **self.kwargs)
        X = add_bias(X, self.bias)
        return self.activation_funct(X, self.weights)

    def predict(self, X):
        X = add_bias(X, self.bias)
        return self.activation_funct(X, self.weights)
