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
