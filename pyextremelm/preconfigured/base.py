# -*- coding: utf-8 -*-
"""
Created on 08.08.16

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
from .. import builder as ELM

class ELMPreConfigured(object):
    def __init__(self):
        self.elm = ELM.ExtremeLearningMachine()

    def fit(self, X, y=None):
        return self.elm.fit(X, y)

    def predict(self, X):
        return self.elm.predict(X)

    def update(self, X, y=None, decay=1):
        return self.elm.update(X, y, decay)

    def fit_batch(self, X, y=None):
        return self.elm.fit_batch(X, y)

    def print_network_structure(self):
        return self.elm.print_network_structure()
