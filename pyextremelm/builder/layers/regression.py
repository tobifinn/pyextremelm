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


class ELMNaive(ELMLayer):
    def __init__(self):
        super().__init__(n_features=1, activation='linear', bias=False)
        self.hidden_matrices = {'K': np.empty((0,0)), 'A': np.empty((0,0))}

    def _calc_weights(self):
        K = self.hidden_matrices['K']
        A = self.hidden_matrices['A']
        self.weights['input'] = np.linalg.inv(K).dot(A)
        self.activation_fct.weights = self.weights

    def fit(self, X, y=None):
        self.n_features=y.shape[1]
        self.hidden_matrices['K'] = X.T.dot(X)
        self.hidden_matrices['A'] = X.T.dot(y)
        self._calc_weights()
        self.activation_fct.weights = self.weights
        return self.predict(X)

    def update(self, X, y, decay=1):
        self.hidden_matrices['K'] += decay.dot(X.T.dot(X))
        self.hidden_matrices['A'] += decay.dot(X.T.dot(y))
        self._calc_weights()
        return self.predict(X)


class ELMRidge(ELMLayer):
    def __init__(self, C=100):
        super().__init__(n_features=1, activation='linear', bias=False)
        self._C = C
        self.hidden_matrices = {'K': np.empty((0,0)), 'A': np.empty((0,0))}

    def __str__(self):
        s = super().__str__()[:-1]
        s += ", L2-constrain: {0:s})".format(str(self._C))
        return s

    def _calc_weights(self):
        K = self.hidden_matrices['K']
        A = self.hidden_matrices['A']
        self.weights['input'] = np.linalg.inv(
            K+np.eye(K.shape[0])/self._C).dot(A)
        self.activation_fct.weights = self.weights

    def fit(self, X, y=None):
        self.n_features=y.shape[1]
        self.hidden_matrices['K'] = X.T.dot(X)
        self.hidden_matrices['A'] = X.T.dot(y)
        self._calc_weights()
        self.activation_fct.weights = self.weights
        return self.predict(X)

    def update(self, X, y, decay=1):
        self.hidden_matrices['K'] += decay.dot(X.T.dot(X))
        self.hidden_matrices['A'] += decay.dot(X.T.dot(y))
        self._calc_weights()
        return self.predict(X)



