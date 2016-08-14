# -*- coding: utf-8 -*-
"""
Created on 14.08.16

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
import theano
import theano.tensor as T
import theano.tensor.signal.pool as Pool

# Internal modules
from .base import ELMConvLayer

class ELMPool(ELMConvLayer):
    """
    The ELMConvolutional represents one convolutional layer within an ELM.

    Attributes:
        n_features (int): Number of features within the layer.
        spatial_extent (tuple[int]): The spatial extent of the
            local receptive field in pixels.
            Size should be (width, height, optional[channels]). If channels
            isn't set all available channels of the input will be used.
        stride (int):The stride of the local receptive fields in pixels.
        zero_padding (int): The zero-padding of the input layer in pixels.
        activation_funct (str or numpy function):
            The function with which the values should be activated,
            Default is None, because in some layers there is no activation.
    """
    def __init__(self, pooling="max",
                 spatial=(2, 2), stride=None, pad=(0,0),
                 ignore_border=False, activation=None):
        """
        Args:
            n_features (int): Number of features within the layer.
            spatial_extent (tuple[int]): The spatial extent of the
                pooling in pixels. Size should be (height, width).
            stride (int):The stride of the local receptive fields in pixels.
            zero_padding (int): The zero-padding of the input layer in pixels.
            activation (optional[str or activation function]):
                The function with which the values should be activated,
                Default is a linear activation.
        """
        if (not hasattr(pad, '__iter__')) and (not pad is None)\
                and (pad=='same'):
            zero_padding = (pad, pad)
        super().__init__(1, spatial, stride, pad, activation, False)
        self.pooling = pooling
        self.ignore_border = ignore_border

    def _generate_conv(self):
        input = T.tensor4(name='input')
        if self.pooling == 'squareroot':
            conv_out = Pool.pool_2d(
                T.power(input,2),
                ds=(self.spatial[0], self.spatial[1]),
                ignore_border=self.ignore_border,
                mode='sum',
                padding=self.pad,
                st=None if self.stride is None else (self.stride, self.stride))
            conv_out = T.sqrt(conv_out)
        else:
            conv_out = Pool.pool_2d(
                input,
                ds=(self.spatial[0], self.spatial[1]),
                ignore_border=self.ignore_border,
                mode=self.pooling,
                padding=self.pad,
                st=None if self.stride is None else (self.stride, self.stride))
        if self.activation_fct is None:
            output = conv_out
        else:
            output = self.activation_fct(conv_out)
        self.conv = theano.function([input], output)

    def fit(self, X, y=None):
        self._generate_conv()
        return self.predict(X)

    def update(self, X, y=None, decay=1):
        return self.predict(X)

    def predict(self, X):
        return self.conv(X)
