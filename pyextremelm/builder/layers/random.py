# -*- coding: utf-8 -*-
"""
Created on 20.05.16

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
from ..base import ELMLayer

__version__ = "0.1"


class ELMRandom(ELMLayer):
    def __init__(self, n_neurons, activation="sigmoid", bias=True, rng=None):
        super().__init__(n_neurons, activation, bias)
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


class ELMOrthoRandom(ELMLayer):
    def __init__(self, n_neurons, activation="sigmoid", bias=True, rng=None):
        super().__init__(n_neurons, activation, bias)
        self.rng = rng
        if self.rng is None:
           self.rng = np.random.RandomState(42)

    def train_algorithm(self, X, y):
        weights = {"input": None, "bias": None}
        input_weights = self.rng.randn(self.get_dim(X), self.n_neurons)
        if self.get_dim(X) > self.n_neurons:
            weights["input"] = scipy.linalg.orth(input_weights)
        else:
            weights["input"] = scipy.linalg.orth(input_weights.T).T
        if self.bias:
            weights["bias"] = np.linalg.qr(self.rng.randn(1, self.n_neurons).T
                                           )[0].T
        return weights
