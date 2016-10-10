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
import abc

# External modules
import numpy as np

# Internal modules


class ELMClass(object):
    def __init__(self, labels=[]):
        """
        Layer class to convert the regression into a classification.
        Args:
            labels:
        """
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

    def update(self, X, y=None, decay=1):
        label, probs, X = self.predict(X)
        return probs, label, X

    @abc.abstractmethod
    def predict(self, X):
        pass


class ELMMultiClass(ELMClass):
    def predict(self, X):
        label = X.argmax(axis=1)
        return label, X


class ELMSoftMax(ELMClass):
    def predict(self, X):
        probs = (np.exp(X).T / np.sum(np.exp(X), axis=1)).T
        label = np.array([self.labels[l] for l in probs.argmax(axis=1)])
        return label, probs, X


class ELMSingleClass(ELMClass):
    def predict(self, X):
        label = X>0
        return label, X