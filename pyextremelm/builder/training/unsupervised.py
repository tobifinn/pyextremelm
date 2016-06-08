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

# Internal modules
from ..helpers import add_bias
from ..activations import sigmoid
from .random import ELMOrthoRandom
from .supervised import ELMRidge, ELMLasso

__version__ = "0.1"


def ELMAE(X=None, y=None, n_neurons=1, bias=True, C=0):
    X = add_bias(X, bias)
    input_weights = ELMOrthoRandom(X=X["input"], y=y, n_neurons=n_neurons, bias=bias)
    G = sigmoid(X, input_weights)
    weights = ELMRidge(X=G, y=X["input"], C=C)
    return {"input": weights["input"].T, "bias": None}


def ELMSparseAE(X=None, y=None, n_neurons=1, bias=True, C=0):
    X = add_bias(X, bias)
    input_weights = ELMOrthoRandom(X=X["input"], y=y, n_neurons=n_neurons, bias=bias)
    G = sigmoid(X, input_weights)
    weights = ELMLasso(X=G, y=X["input"], C=C)
    weights["input"] = weights["input"].reshape((weights["input"].shape[0], 1))
    return {"input": weights["input"].T, "bias": None}


def USELM(X=None, y=None, n_neurons=1, bias=True, C=0):
    pass