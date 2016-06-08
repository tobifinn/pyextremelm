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
from .. import helpers

__version__ = "0.1"


def ELMRandom(X=None, y=None, n_neurons=1, bias=False):
    weights = {"input": None, "bias": None}
    weights["input"] = np.random.randn(helpers.get_dim(X), n_neurons)
    if bias:
        weights["bias"] = np.random.randn(1, n_neurons)
    return weights

def ELMOrthoRandom(X=None, y=None, n_neurons=1, bias=False):
    weights = {"input": None, "bias": None}
    input_weights = np.random.randn(helpers.get_dim(X), n_neurons)
    if helpers.get_dim(X)>n_neurons:
        weights["input"] = scipy.linalg.orth(input_weights)
    else:
        weights["input"] = scipy.linalg.orth(input_weights.T).T
    if bias:
        weights["bias"] = np.linalg.qr(np.random.randn(1, n_neurons).T)[0].T
    return weights
