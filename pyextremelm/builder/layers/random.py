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

# External modules
import numpy as np

# Internal modules
from .base import ELMLayer


class ELMRandom(ELMLayer):
    """
    This layer represents a random layer within the neural network
    """
    def __init__(self, n_features, activation="sigmoid", ortho=False,
                 bias=True, rng=None):
        super().__init__(bias)
        self.n_features = n_features
        self.activation = activation
        self.ortho = ortho
        self.rng = rng
        if self.rng is None:
           self.rng = np.random.RandomState(42)

    def train_algorithm(self, X, y):
        weights = {
            "input": self.rng.randn(self.get_dim(X), self.n_neurons),
            "bias": None}
        if self.bias:
            weights["bias"] = self.rng.randn(1, self.n_neurons)
        return weights

    def fit(self, X, y=None):
        self.weights = self.train_algorithm(X, y)
        X = self.add_bias(X)
        self.activation_funct = self.activation_funct(self.weights)
        return self.activation_funct.activate(X)

    def predict(self, X, **kwargs):
        X = self.add_bias(X)
        return self.activation_funct.activate(X)

    def update(self, X, y=None, decay=1):
        return self.predict(X)

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

