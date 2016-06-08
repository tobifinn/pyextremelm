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
from sklearn.base import clone

# Internal modules
from pyextremelm.builder import base as ELM
from pyextremelm.builder import metrics, layers as ELMLayers
#from .supervised import ELMSupervised

__version__ = "0.1"


class ELMRegressor(object):
    def __init__(self, hidden_neurons, activation="sigmoid", C=0, iters=1):
        self.elm = ELM.ExtremeLearningMachine(metrics.MeanSquaredError, iters)
        self.elm.add_existing_layer(
            ELMLayers.ELMRandom(hidden_neurons, activation=activation))
        self.elm.add_existing_layer(ELMLayers.ELMRidge(C=C))

    def fit(self, X, y):
        self.elm.fit(X, y)

    def predict(self, X):
        return self.elm.predict(X)

