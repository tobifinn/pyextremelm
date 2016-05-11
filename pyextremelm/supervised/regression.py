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

# Internal modules
from .supervised import ELMSupervised

__version__ = "0.1"


class ELMRegressor(ELMSupervised):
    def _train(self, X, y):
        weights = []
        accuracy = []
        for i in range(self.rand_iter):
            weights.append(np.random.randn(
                X.shape[1] if len(X.shape)>1 else 1,
                self.n_hidden_neurons))
            #print(X.shape, weights[i].shape, y.shape)
            G = self.activation_function(X.dot(weights[i]))
            output_weights = np.linalg.pinv(G).dot(y)
            accuracy.append(np.mean((G.dot(output_weights)-y)**2))
        if self.rand_select == "best":
            best_key = accuracy.index(min(accuracy))
        self.random_weights = weights[best_key]
        G = self.activation_function(X.dot(self.random_weights))
        self.output_weights = np.linalg.pinv(G).dot(y)

    def predict(self, X):
        if self.bias:
            if len(X.shape) > 1:
                X = np.column_stack([X, np.ones([X.shape[0], 1])])
            else:
                X = np.column_stack(np.append(X, 1))
        G = self.activation_function(X.dot(self.random_weights))
        return G.dot(self.output_weights)
