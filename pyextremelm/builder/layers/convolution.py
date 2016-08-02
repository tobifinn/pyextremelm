# -*- coding: utf-8 -*-
"""
Created on 23.06.16

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
import abc
import warnings

# External modules
import numpy as np
import scipy

import theano
import theano.tensor as T
import theano.tensor.nnet.neighbours
import theano.tensor.signal.pool as Pool

# Internal modules
from ..base import ELMLayer
from .supervised import ELMRidge
from .random import ELMOrthoRandom
from .shape import PadLayer

__version__ = "0.1"

named_activations = {
    "sigmoid": T.nnet.sigmoid,
    "hardsig": T.nnet.hard_sigmoid,
    "tanh": T.tanh,
    "fourier": T.cos,
    "relu": T.nnet.relu,
    "linear": None
}

unnamed_activations = []


class BaseConv(ELMLayer):
    """
    The BaseConv represents a basic convolutional layer within an ELM.

    Attributes:
        n_features (int): Number of features within the layer.
        spatial_extent (tuple[int]): The spatial extent of the
            local receptive field in pixels.
            Size should be (width, height, optional[channels]). If channels
            isn't set all available channels of the input will be used.
        stride (int):The stride of the local receptive fields in pixels.
        zero_padding (int): The zero-padding of the input layer in pixels.
        train_algorithm (Child of ELMTraining): Training method of the layer.
        activation_funct (str or numpy function):
            The function with which the values should be activated,
            Default is None, because in some layers there is no activation.

    Args:
        n_features (int): Number of features within the layer.
        spatial_extent (tuple[int]): The spatial extent of the
            local receptive field in pixels.
            Size should be (channels, height, width).
        stride (int):The stride of the local receptive fields in pixels.
        zero_padding (int): The zero-padding of the input layer in pixels.
        activation (optional[str or activation function]):
            The function with which the values should be activated,
            Default is a linear activation.
        bias (bool): If the layer should have a bias. Default is False.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, n_features, spatial_extent=(3, 5, 5), stride=1,
                 zero_padding='valid', activation="relu", bias=False,
                 rng=None):
        self.filter_shape = tuple([n_features] + list(spatial_extent))
        self.spatial_extent = spatial_extent
        self.stride = stride
        self.zero_padding = zero_padding
        self.conv = None
        self.rng = rng
        self.activations = named_activations
        super().__init__(n_features, activation, bias)
        if self.rng is None:
            self.rng = np.random.RandomState(42)

    def __str__(self):
        s = "{0:s}(filters shape: {1:s}, stride: {2:d}," \
            "padding: {3:s}, activation: {4:s}, bias: {5:s})".format(
            self.__class__.__name__, str(self.filter_shape), self.stride,
            str(self.zero_padding), str(type(self.activation_funct).__name__),
            str(self.bias))
        return s

    @property
    def trained(self):
        """
        Property to check if the layer is trained.
        Returns:
            trained (bool): True if the layer is trained, else False.
        """
        if self.weights is None:
            trained = False
        else:
            trained = True
        return trained

    @abc.abstractmethod
    def fit(self, X, y=None):
        pass

    @abc.abstractmethod
    def _generate_conv(self):
        pass

    def predict(self, X):
        return self.conv(X)

    def get_activation(self, funct="sigmoid"):
        """
        Function to get the activation function
        Args:
            funct (str or function):

        Returns:
            function: The activation function
        """
        if isinstance(funct, str) and funct in self.activations:
            return named_activations[funct]
        else:
            return funct


class ELMLRF(BaseConv):
    """
    The ELMLRF represents a local receptive field layer within an ELM [1].

    Attributes:
        n_features (int): Number of features within the layer.
        spatial_extent (tuple[int]): The spatial extent of the
            local receptive field in pixels.
            Size should be (width, height, optional[channels]). If channels
            isn't set all available channels of the input will be used.
        stride (int):The stride of the local receptive fields in pixels.
        zero_padding (int): The zero-padding of the input layer in pixels.
        train_algorithm (Child of ELMTraining): Training method of the layer.
        activation_funct (str or numpy function):
            The function with which the values should be activated,
            Default is None, because in some layers there is no activation.
    -----------------------------------------------------------------------
    [1] Huang, Guang-Bin, et al. "Local receptive fields based extreme learning
        machine." IEEE Computational Intelligence Magazine 10.2 (2015): 18-29.
    """
    def __init__(self, n_features, spatial_extent=(3, 5, 5), stride=1,
                 zero_padding='valid', activation="relu", bias=True,
                 rng=None):
        """
        Args:
            n_features (int): Number of features within the layer.
            spatial_extent (tuple[int]): The spatial extent of the
                local receptive field in pixels.
                Size should be (channels, height, width).
            stride (int):The stride of the local receptive fields in pixels.
            zero_padding (int): The zero-padding of the input layer in pixels.
            activation (optional[str or activation function]):
                The function with which the values should be activated,
                Default is a linear activation.
            bias (bool): If the layer should have a bias. Default is False.
        """
        self.activations = named_activations
        super().__init__(n_features, spatial_extent, stride, zero_padding,
                         activation, bias, rng)

    def _generate_conv(self, image_shape=None):
        input = T.tensor4(name='input')
        W = theano.shared(np.asarray(self.weights['input'], dtype=input.dtype),
                          name='W')
        conv_out = T.nnet.conv2d(input, W,
                                 border_mode=self.zero_padding,
                                 subsample=(self.stride, self.stride),
                                 filter_shape=self.filter_shape,
                                 input_shape=image_shape)
        if self.bias:
            b = theano.shared(
                np.asarray(self.weights['bias'], dtype=input.dtype),
                name='b')
            conv_out = conv_out + b.dimshuffle('x', 0, 'x', 'x')
        if self.activation_funct is None:
            output = conv_out
        elif self.activation_funct == "hardlimit":
            output = conv_out>0
        elif self.activation_funct == "hardtanh":
            output = T.switch(conv_out > -1, T.switch(conv_out > 1, 1, conv_out), -1)
        else:
            output = self.activation_funct(conv_out)
        self.conv = theano.function([input], output)

    def train_algorithm(self, X, y):
        weights = {"input": None, "bias": None}
        weights["input"] = self.rng.standard_normal(size=self.filter_shape)
        if self.bias:
            weights["bias"] = self.rng.standard_normal(size=(self.n_neurons,))
        return weights

    def fit(self, X, y=None):
        self.weights = self.train_algorithm(X, y)
        self._generate_conv((None, X.shape[1], X.shape[2], X.shape[3]))
        return self.predict(X)


class ELMPool(BaseConv):
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
    def __init__(self, pooling="max", ignore_border=False,
                 spatial_extent=(2, 2), stride=None, zero_padding=(0,0),
                 activation=None):
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
        self.activations = named_activations
        if (not hasattr(zero_padding, '__iter__')) and (not zero_padding is None)\
                and (zero_padding=='same'):
            zero_padding = (zero_padding, zero_padding)
        super().__init__(1, spatial_extent, stride, zero_padding, activation)
        self.pooling = pooling
        self.ignore_border = ignore_border

    def _generate_conv(self):
        input = T.tensor4(name='input')
        if self.pooling == 'squareroot':
            conv_out = Pool.pool_2d(
                T.power(input,2),
                ds=(self.spatial_extent[0], self.spatial_extent[1]),
                ignore_border=self.ignore_border,
                mode='sum',
                padding=self.zero_padding,
                st=None if self.stride is None else (self.stride, self.stride))
            conv_out = T.sqrt(conv_out)
        else:
            conv_out = Pool.pool_2d(
                input,
                ds=(self.spatial_extent[0], self.spatial_extent[1]),
                ignore_border=self.ignore_border,
                mode=self.pooling,
                padding=self.zero_padding,
                st=None if self.stride is None else (self.stride, self.stride))
        if self.activation_funct is None:
            output = conv_out
        else:
            output = self.activation_funct(conv_out)
        self.conv = theano.function([input], output)

    def fit(self, X, y=None):
        self._generate_conv()
        return self.predict(X)


class ELMConvAE_linear(ELMLRF):
    def __init__(self, n_features, spatial_extent=(5, 5), stride=1,
                 zero_padding=0, activation="relu", bias=False,
                 rng=None, C=0):
        """
        Args:
            n_features (int): Number of features within the layer.
            spatial_extent (tuple[int]): The spatial extent of the
                local receptive field in pixels.
                Size should be (channels, height, width).
            stride (int):The stride of the local receptive fields in pixels.
            zero_padding (int): The zero-padding of the input layer in pixels.
            activation (optional[str or activation function]):
                The function with which the values should be activated,
                Default is a linear activation.
            bias (bool): If the layer should have a bias. Default is False.
            C (optional[float]): The constrain factor for the regression.
        """
        self.activations = named_activations
        super().__init__(n_features, spatial_extent, stride, zero_padding,
                         activation, False, rng)
        #self.neib_shape = tuple([1] + list(spatial_extent))
        self.img2neib = None
        self._generate_img2neib()
        self.layers = [ELMOrthoRandom(n_features, activation, bias),
                       ELMRidge(C)]

    def _generate_img2neib(self):
        input = T.tensor4(name='input')
        output = T.nnet.neighbours.images2neibs(
            input, neib_shape=self.spatial_extent,
            neib_step=None if self.stride is None else (self.stride, self.stride))
        self.img2neib = theano.function([input], output)

    def _add_pad(self, X):
        if isinstance(self.zero_padding, int) and (self.zero_padding>0):
            layer = PadLayer(self.zero_padding)
            return layer.fit(X)
        else:
            return X

    def train_algorithm(self, X, y=None):
        x = self.img2neib(X)
        x = x.reshape((X.shape[0], X.shape[1], -1, self.spatial_extent[0],
                       self.spatial_extent[1])).transpose(0,2,3,4,1)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = x.reshape((-1, x.shape[-1]))
        layer_input = x
        for layer in self.layers:
            layer_input = layer.fit(layer_input, x)
        weights = self.layers[-1].weights
        if len(weights["input"].shape)<2:
            weights["input"] = weights["input"].reshape(-1, 1)
        spatial = tuple([weights["input"].shape[0]] +
                        [X.shape[1]] +
                        list(self.spatial_extent))
        weights["input"] = weights["input"].reshape(spatial)
        return weights

    def fit(self, X, y=None):
        X = self._add_pad(X)
        self.weights = self.train_algorithm(X, y)
        self._generate_conv((None, X.shape[1], X.shape[2], X.shape[3]))
        return self.predict(X)

#    def fit(self, X, y=None):
#        return super().fit(X, y)

    def predict(self, X):
        X = self._add_pad(X)
        return super().predict(X)