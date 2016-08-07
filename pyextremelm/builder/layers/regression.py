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

# Internal modules
from .base import ELMLayer


class ELMRidge(ELMLayer):
    def __init__(self, C=0):
        super().__init__(n_features=None, activation='linear', bias=False)
        self.C = C
        self.hidden_matrices = {'K': None, 'A': None}

    def __str__(self):
        s = super().__str__()
        s += "L2-constrain: {0:s})".format(str(self.__C))
        return s

    def fit(self, X, y=None):

