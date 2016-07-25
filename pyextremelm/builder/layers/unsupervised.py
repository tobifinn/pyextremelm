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

# Internal modules
from ..base import ELMLayer
from .random import ELMOrthoRandom
from .supervised import ELMRidge, ELMLasso

__version__ = "0.1"


class ELMAE(ELMLayer):
    def __init__(self, n_neurons, activation="sigmoid", bias=True, C=0):
        super().__init__(n_neurons, activation, bias)
        self.C = C
        self.layers = [ELMOrthoRandom(n_neurons, activation, bias),
                       ELMRidge(C)]

    def __str__(self):
        s = "{0:s}(neurons: {1:d}, activation: {2:s}, bias: {3:s}, " \
            "C: {4:s}, sublayers: {5:s})".format(
            self.__class__.__name__, self.n_neurons,
            str(type(self.activation_funct).__name__), str(self.bias),
            str(self.C), str([type(l).__name__ for l in self.layers]))
        return s

    def train_algorithm(self, X, y=None):
        layer_input = X
        for layer in self.layers:
            layer_input = layer.fit(layer_input, X)
        weights = self.layers[-1].weights
        if len(weights["input"].shape)<2:
            weights["input"] = weights["input"].reshape(-1, 1)
        weights["input"] = weights["input"].T
        return weights


class ELMSparseAE(ELMAE):
    def __init__(self, n_neurons, activation="sigmoid", bias=True, C=0):
        super().__init__(n_neurons, activation, bias, C=C)
        self.layers = [ELMOrthoRandom(n_neurons, activation, bias),
                       ELMLasso(C)]
#
#
# class ELMLRF(ELMLayer):
#     """
#     The ELMConvolutional represents one convolutional layer within an ELM.
#
#     Attributes:
#         n_features (int): Number of features within the layer.
#         spatial_extent (tuple[int]): The spatial extent of the
#             local receptive field in pixels.
#             Size should be (width, height, optional[channels]). If channels
#             isn't set all available channels of the input will be used.
#         stride (int):The stride of the local receptive fields in pixels.
#         zero_padding (int): The zero-padding of the input layer in pixels.
#         train_algorithm (Child of ELMTraining): Training method of the layer.
#         activation_funct (str or numpy function):
#             The function with which the values should be activated,
#             Default is None, because in some layers there is no activation.
#     """
#     def __init__(self, n_features, spatial_extent, zero_padding):
#         """
#         Args:
#             n_features (int): Number of features within the layer.
#             spatial_extent (tuple[int]): The spatial extent of the
#                 local receptive field in pixels.
#                 Size should be (width, height, optional[channels]). If channels
#                 isn't set all available channels of the input will be used.
#             stride (int):The stride of the local receptive fields in pixels.
#             zero_padding (int): The zero-padding of the input layer in pixels.
#             train_algorithm (function from training directory):
#                 Training algorithm of the layer.
#             activation_funct (optional[str or activation function]):
#                 The function with which the values should be activated,
#                 Default is a linear activation.
#             bias (bool): If the layer should have a bias. Default is False.
#         """
#         self.n_features = n_features
#         self.spatial_extent = spatial_extent
#         self.n_neurons = np.sum(self.spatial_extent*n_features)
#         self.zero_padding = zero_padding
#         self.weights = None
#
#     @property
#     def trained(self):
#         """
#         Property to check if the layer is trained.
#         Returns:
#             trained (bool): True if the layer is trained, else False.
#         """
#         if self.weights is None:
#             trained = False
#         else:
#             trained = True
#         return trained
#
#     def fit(self, X, y=None, **kwargs):
#         self.weights = ELMTraining.ELMRandom(X, y, self.n_neurons, False).\
#             reshape(self.spatial_extent+(self.n_features,))
#         return self.activation_funct(X, self.weights, **kwargs)
#
#     def predict(self, X, **kwargs):
#         return self.activation_funct(X, self.weights, **kwargs)
