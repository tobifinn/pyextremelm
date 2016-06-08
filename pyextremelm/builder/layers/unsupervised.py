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
from ..base import ELMLayer
import pyextremelm.builder.training as ELMTraining

__version__ = "0.1"


class ELMAE(ELMLayer):
    def __init__(self, n_neurons, activation="sigmoid", C=10E10):
        super().__init__(n_neurons, ELMTraining.ELMAE,
                         activation, True, C=C)


class ELMSparseAE(ELMLayer):
    def __init__(self, n_neurons, activation="sigmoid", C=1):
        super().__init__(n_neurons, ELMTraining.ELMSparseAE,
                         activation, True, C=C)
