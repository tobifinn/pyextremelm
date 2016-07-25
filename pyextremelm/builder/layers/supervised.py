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

from sklearn.linear_model import Lasso

# Internal modules
from ..base import ELMLayer

__version__ = "0.1"


class ELMRidge(ELMLayer):
    def __init__(self, C=0):
        super().__init__(1, 'linear', False)
        self.__C = C
        self.K = None

    def __str__(self):
        s = "{0:s}(neurons: {1:d}, activation: {2:s}, bias: {3:s}, " \
            "C: {4:s})".format(
            self.__class__.__name__, self.n_neurons,
            str(type(self.activation_funct).__name__), str(self.bias), str(self.__C))
        return s

    def update(self, X, y):
        try:
            self.K += X.T.dot(X)
            self.weights["input"] += np.linalg.inv(self.K).dot(X.T).dot(
                y-X*self.weights["input"])
        except:
            self.K += X.dot(X.T)
            self.weights["input"] += X.T.dot(np.linalg.inv(self.K)).dot(
                y-X*self.weights["input"])

    def train_algorithm(self, X, y):
        self.n_neurons = y.shape[1]
        if self.__C > 0:
            try:
                self.K = X.T.dot(X) + np.eye(X.shape[1]) / self.__C
                factors = np.linalg.inv(self.K).dot(X.T).dot(y)
            except:
                self.K = X.dot(X.T) + np.eye(X.shape[1]) / self.__C
                factors = X.T.dot(np.linalg.inv(self.K)).dot(y)
        else:
            self.K = X.T.dot(X)
            factors = np.linalg.pinv(X).dot(y)
        return {"input": factors, "bias": None}


class ELMLasso(ELMLayer):
    def __init__(self, C=0):
        super().__init__(1, 'linear', False)
        self.C = C

    def __str__(self):
        s = "{0:s}(neurons: {1:d}, activation: {2:s}, bias: {3:s}, " \
            "C: {4:s})".format(
            self.__class__.__name__, self.n_neurons,
            str(type(self.activation_funct).__name__), str(self.bias), str(self.C))
        return s

    def train_algorithm(self, X, y):
        self.n_neurons = y.shape[1]
        if self.C > 0:
            factors = Lasso(alpha=1 / self.C, fit_intercept=False).fit(
                X, y).coef_
        else:
            factors = np.linalg.pinv(X).dot(y)
        return {"input": factors.T, "bias": None}


class ELMClass(object):
    def __init__(self, labels=[]):
        self.labels = labels

    def __str__(self):
        s = "{0:s}(labels: {1:s}, n_labels: {2:d})".format(
            self.__class__.__name__, str(self.labels), self.n_labels)
        return s

    @property
    def n_labels(self):
        if not self.labels is None:
            return len(self.labels)
        else:
            return 0

    def labels_bin(self, X):
        unique = list(np.unique(X))
        for u in unique:
            if u not in self.labels:
                self.labels.append(u)
        binarized = None
        for l in self.labels:
            labelized = (X == l).astype(int).reshape((-1, 1))
            if binarized is None:
                binarized = labelized
            else:
                binarized = np.c_[binarized, labelized]
        #print(binarized.shape)
        return binarized

    def fit(self, X, y=None):
        label, probs, X = self.predict(X)
        return probs, label, X

    def predict(self, X):
        probs = (np.exp(X).T / np.sum(np.exp(X), axis=1)).T
        label = np.array([self.labels[l] for l in probs.argmax(axis=1)])
        return label, probs, X
