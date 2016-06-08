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
from ..training import supervised

__version__ = "0.1"


class ELMRegression(ELMLayer):
    def __init__(self, train_algorithm, **kwargs):
        super().__init__(1, train_algorithm, "linear", False, **kwargs)


class ELMRidge(ELMRegression):
    def __init__(self, C=0):
        super().__init__(supervised.ELMRidge, C=C)


class ELMLasso(ELMRegression):
    def __init__(self, C=0):
        super().__init__(supervised.ELMLasso, C=C)


class ELMClassification(ELMLayer):
    def __init__(self, output_neurons, train_algorithm, **kwargs):
        super().__init__(output_neurons, train_algorithm, "linear", False,
                         **kwargs)
