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

# Internal modules
from .base import ELMLayer
from .random import ELMRandom
from .regression import ELMRegression


class ELMAE(ELMLayer):
    def __init__(self, n_features, activation="sigmoid", bias=True, C=0,
                 ortho=True, rng=None):
        super().__init__(n_features, activation, bias)
        self.layers = [ELMRandom(n_features, activation, bias, ortho, rng),
                       ELMRegression(C)]
        self._C = C
        self.ortho = ortho

    def __str__(self):
        s = "{0:s}, orthogonalized: {1:s}, L2-constrain: {2:f})".format(
            super().__str__()[:-1], str(self.ortho), self._C)
        return s

    def train_algorithm(self, X, decay=False):
        layer_input = X
        for layer in self.layers:
            if not isinstance(decay, bool):
                layer_input = layer.update(layer_input, X, decay)
            else:
                layer_input = layer.fit(layer_input, X)
        weights = self.layers[-1].weights
        if len(weights["input"].shape)<2:
            weights["input"] = weights["input"].reshape(-1, 1)
        return weights

    def fit(self, X, y=None):
        self.weights = self.train_algorithm(X)
        self.weights['input'] = self.weights['input'].T
        try:
            self.activation_fct.weights = self.weights
        except Exception as e:
            raise ValueError('This activation isn\'t implemented yet'
                             '\n(original exception: {0:s})'.format(e))
        return self.predict(X)

    def update(self, X, y=None, decay=1):
        self.weights = self.train_algorithm(X, decay)
        self.weights['input'] = self.weights['input'].T
        try:
            self.activation_fct.weights = self.weights
        except Exception as e:
            raise ValueError('This activation isn\'t implemented yet'
                             '\n(original exception: {0:s})'.format(e))
        return self.predict(X)
