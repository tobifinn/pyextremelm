# -*- coding: utf-8 -*-
"""
Created on 13.07.16

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
import warnings

# External modules
import numpy as np

# Internal modules

__version__ = "0.1"


class ShapeLayer(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def algorithm(self, X):
        pass

    def fit(self, X, y=None):
        return self.algorithm(X)

    def update(self, X, y=None, decay=1):
        return self.algorithm(X)

    def predict(self, X):
        return self.algorithm(X)


class FlattenLayer(ShapeLayer):
    """
    Object to flatten a m dim matrix into a n dim matrix, so that n<m.

    Args:
        retain_dim (optional[int]): How many first dimensions should be
            retained. Default is 1. A number below 1 means that no dimension
            will be retained.

    """
    def __init__(self, retain_dims=1):
        if isinstance(retain_dims, int) and retain_dims>0:
            self.retain_dims = retain_dims
        else:
            self.retain_dims = 0

    def algorithm(self, X):
        if len(X.shape)<self.retain_dims:
            warnings.warn("The input shape is smaller than the dimensions which should be retained", UserWarning)
            return X
        else:
            shape = []
            if self.retain_dims>0:
                shape = [X.shape[i] for i in range(self.retain_dims)]
            shape.append(-1)
            return X.reshape(tuple(shape))


class PadLayer(ShapeLayer):
    """
    Object which appends a padding to the matrix
    """
    def __init__(self, width, val=0, retain_dims=2):
        self.width = width
        self.val = val
        self.retain_dims = retain_dims

    def algorithm(self, X):
        if len(X.shape)<self.retain_dims:
            warnings.warn("The input shape is smaller than the dimensions which should be retained", UserWarning)
        else:
            shape = [(0,0)]*self.retain_dims
            shape += [(self.width, self.width)]*(len(X.shape)-self.retain_dims)
            X = np.pad(X, tuple(shape), mode='constant', constant_values=0)
        return X
