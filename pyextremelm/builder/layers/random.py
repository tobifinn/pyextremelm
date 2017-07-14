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
import scipy

# Internal modules
from .base import ELMLayer
from pyextremelm.builder.activations.dense import Linear


class ELMRandom(ELMLayer):
    """
    This layer represents a random layer within the neural network
    """
    def __init__(self, n_features, activation="sigmoid", bias=True,
                 ortho=False, rng=None,):
        super().__init__(n_features, activation, bias)
        self.ortho = ortho
        self.rng = rng
        if self.rng is None:
           self.rng = np.random.RandomState(42)

    def __str__(self):
        s = "{0:s}, orthogonalized: {1:s})".format(
            super().__str__()[:-1], str(self.ortho))
        return s

    def train_algorithm(self, X, y=None):
        weights = {
            "input": self.rng.randn(self.get_dim(X), self.n_features),
            "bias": None}
        if self.bias:
            weights["bias"] = self.rng.randn(1, self.n_features)
        if self.ortho:
            if self.get_dim(X) > self.n_features:
                weights["input"] = scipy.linalg.orth(weights["input"])
            else:
                weights["input"] = scipy.linalg.orth(weights["input"].T).T
            if self.bias:
                weights["bias"] = np.linalg.qr(weights["bias"].T)[0].T
        return weights

    def fit(self, X, y=None):
        self.weights = self.train_algorithm(X, y)
        try:
            self.activation_fct.weights = self.weights
        except Exception as e:
            raise ValueError('This activation isn\'t implemented yet'
                  '\n(original exception: {0:s})'.format(e))
        return self.predict(X)

    def update(self, X, y=None, decay=1):
        return self.predict(X)


class ELMRecurrent(ELMRandom):
    """
    This layer is inspired by the so called echo state network and represents a
    random recurrent layer within the extreme learning machine. The input
    should be normalized.
    """
    def __init__(self, n_features, activation="sigmoid", bias=True,
                 ortho=False, rng=None,):
        super().__init__(n_features, activation, bias, ortho, rng)
        self.state = None
        self.lin_activation = Linear()
        self.decay = 1

    def update_state(self, X):
        len_x, _ = X.shape
        X = self.add_bias(X)
        return_state = np.zeros((len_x, self.n_features))
        for i in range(len_x):
            return_state[i] = self._single_state_update(X[i, :])
        return return_state
    
    def _single_state_update(self, x):
        state_update = self.lin_activation.activate(x)
        if self.state is None:
            self.state = state_update
        else:
            self.state = self.decay*self.state + state_update
        return self.state

    def fit(self, X, y=None):
        self.weights = self.train_algorithm(X, y)
        self.lin_activation.weights = self.weights
        return self.update(X)

    def update(self, X, y=None, decay=1):
        self.decay = decay
        updated_state = self.update_state(X)
        return self._predict_update(X, updated_state)

    def _predict_updated(self, X, updated_state=None):
        if updated_state is None:
            len_x, _ = X.shape
            updated_state = self.state.reshape((1, -1))
            updated_state = np.repeat(updated_state, len_x, axis=0)
        self.activation_fct.weights = np.ones((self.n_features, self.n_features))
        activated_state = self.activation_fct.activate(updated_state)
        predicted_state = np.concatenate([activated_state, X], axis=1)
        return predicted_state

    def predict(X):
        return self._predict_update(X, None)

