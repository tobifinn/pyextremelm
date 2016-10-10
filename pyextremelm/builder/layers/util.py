# -*- coding: utf-8 -*-
"""
Created on 24.08.16

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


class ELMNormalize(object):
    def __init__(self, std_norm=0.01):
        self._mean = None
        self._std = None
        self._samples = None
        self._std_norm = std_norm

    def fit(self, X, y=None):
        self._samples = X.shape[0]
        self._mean = np.mean(X, axis=(0,2,3))
        self._std = np.std(X, axis=(0,2,3))
        return self.predict(X)

    def update(self, X, y=None, decay=1):
        new_samples = X.shape[0]
        self._mean = (decay*self._samples*self._mean+new_samples*np.mean(X, axis=(0,2,3)))/(self._samples+new_samples)
        self._std = (decay*self._samples*self._std+new_samples*np.std(X, axis=(0,2,3)))/(self._samples+new_samples)
        self._samples += new_samples
        return self.predict(X)

    def predict(self, X):
        for i in range(X.shape[1]):
            X[:,i,:,:] = X[:,i,:,:]-self._mean[i]
            X[:,i,:,:] = X[:,i,:,:]/(self._std[i]+self._std_norm)
        return X
