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

# Internal modules
from ..base import ELMPreConfigured
from pyextremelm.builder import layers as ELMLayers


class ELMRegressor(ELMPreConfigured):
    def __init__(self, hidden_neurons, activation="sigmoid", C=0):
        """
        A pre-configured class for a regression based extreme learning machine.
        Args:
            hidden_neurons (int): Number of neurons in the hidden layer.
            activation (optional[str]): Activation function for the hidden
                layer. Default is the sigmoid function.
            C (optional[float]): The constrain factor for the regression. If
                there should be no constrain the factor has to be 0.
                Default is no constrain.
        """
        super().__init__()
        self.elm.add_layer(
            ELMLayers.ELMRandom(hidden_neurons, activation=activation))
        self.elm.add_layer(ELMLayers.ELMRegression(C=C))
