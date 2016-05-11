# -*- coding: utf-8 -*-
"""
Created on 06.05.16
Created for pyExtremeLM

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
from sklearn.base import clone

# Internal modules
from ..base import ELMBase

__version__ = "0.1"


class ELMSupervised(ELMBase):
    """
    The abstract base class for a supervised extreme machine learning.
    """
    def fit(self, X, y):
        """
        The supervised training method.

        Args:
            X (np.array): The training input array.
            y (np.array): The training output array.
        """
        if len(X.shape)>1:
            self.n_input_neurons = X.shape[1]
        else:
            self.n_input_neurons = 1
        if len(y.shape)>1:
            self.n_output_neurons = y.shape[1]
        else:
            self.n_output_neurons = 1
        if self.bias:
            X = np.c_[X, np.ones(X.shape[0])]
        self._train(X, y)

    def _train(self, X, y):
        pass

    def predict(self, X):
        pass


class ELMSKSupervised(ELMSupervised):
    def __init__(self, hidden_neurons, sklearn_funct,
                 activation_funct="sigmoid", bias=True,
                 rand_iter = 30, rand_select = "best"):
        super().__init__(hidden_neurons, activation_funct, bias,
                 rand_iter, rand_select)
        self.sklearn_funct = sklearn_funct

    def _train(self, X, y):
        weights = []
        accuracy = []
        for i in range(self.rand_iter):
            weights.append(np.random.randn(
                X.shape[1] if len(X.shape)>1 else 1,
                self.n_hidden_neurons))
            G = self.activation_function(X.dot(weights[i]))
            test_funct = clone(self.sklearn_funct).fit(G, y)
            accuracy.append(np.mean((test_funct.predict(G) - y) ** 2))
        if self.rand_select == "best":
            best_key = accuracy.index(min(accuracy))
        self.random_weights = weights[best_key]
        G = self.activation_function(X.dot(self.random_weights))
        self.output_weights = clone(self.sklearn_funct).fit(G, y)

    def predict(self, X):
        if self.bias:
            X = np.c_[X, np.ones(X.shape[0])]
            # if len(X.shape) > 1:
            #     X = np.column_stack([X, np.ones([X.shape[0], 1])])
            # else:
            #     X = np.column_stack(np.append(X, 1))
        G = self.activation_function(X.dot(self.random_weights))
        return self.output_weights.predict(G)
