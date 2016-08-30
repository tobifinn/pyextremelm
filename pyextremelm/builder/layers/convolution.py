# -*- coding: utf-8 -*-
"""
Created on 07.08.16

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
import sys

# External modules
import numpy as np
import scipy

import theano
from theano.ifelse import ifelse
import theano.tensor as T

# Internal modules
from .base import ELMConvLayer


class ELMLRF(ELMConvLayer):
    def __init__(self, n_features, spatial=(3,3), stride=(1,1), pad=(1,1),
                 activation='linear', bias=True, ortho=False, rng=None):
        """
        The ELMLRF represents a local receptive field layer within an ELM [1].
        This implementation of ELMLRF is based on theano.

        Attributes:
            n_features (int): Number of features within the layer.
            spatial_extent (optional[int/tuple[int]]): The spatial extent of
                the local receptive field in pixels.
                Size should be (width, height) or an integer, so that the width
                and height are the same. Default is a 3x3 lrf.
            stride (optional[int/tuple[int]]):The stride of the local receptive
                fields in pixels. Size should be (width, height) or an integer,
                so that the width and height are the same. Default is a stride
                of 1x1.
            pad (optional[int/tuple[int]]): The zero-padding of the input layer
                in pixels. Size should be (width, height) or an integer,
                so that the width and height are the same. Default is a padding
                of 1x1.
            activation_funct (str or convolutional activation function):
                The function with which the values should be activated,
                Default is None, because in some layers there is no activation.
            ortho (optional[bool]): If the random weights should be
                orthogonalized or not. Default is False.
            rng (optional[numpy RandomState]): A numpy random state to generate
                the random weights. If no state is given, the seed will be set
                to 42.
        -----------------------------------------------------------------------
        [1] Huang, Guang-Bin, et al. "Local receptive fields based extreme
            learning machine." IEEE Computational Intelligence Magazine 10.2
            (2015): 18-29.
        """
        super().__init__(n_features, spatial, stride, pad, activation, bias)
        self.ortho = ortho
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.RandomState(42)

    def _generate_conv(self, image_shape=None):
        input = T.tensor4(name='input')
        W = theano.shared(np.asarray(self.weights['input'], dtype=input.dtype),
                          name='W')
        conv_out = T.nnet.conv2d(input, W,
                                 border_mode=self.pad,
                                 subsample=self.stride,
                                 filter_shape=self.filter_shape,
                                 input_shape=image_shape)
        if self.bias:
            b = theano.shared(
                np.asarray(self.weights['bias'], dtype=input.dtype),
                name='b')
            conv_out = conv_out + b.dimshuffle('x', 0, 'x', 'x')
        if self.activation_fct is None:
            output = conv_out
        elif self.activation_fct == "hardlimit":
            output = conv_out>0
        elif self.activation_fct == "hardtanh":
            output = T.switch(conv_out > -1, T.switch(conv_out > 1, 1, conv_out), -1)
        else:
            output = self.activation_fct(conv_out)
        self.conv_fct = theano.function([input], output)

    def train_algorithm(self, X):
        self.filter_shape = tuple([self.n_features, X.shape[1]]+
                             list(self.spatial))
        weights = {
            "input": self.rng.standard_normal(size=self.filter_shape),
            "bias": None}
        if self.bias:
            weights["bias"] = self.rng.standard_normal(size=(self.n_features,))
        if self.ortho:
            weights['input'] = weights["input"].reshape((self.n_features, -1))
            s = weights['input'].shape
            if s[0] < s[1]:
                weights['input'] = scipy.linalg.orth(weights['input'].T).T
            else:
                weights['input'] = scipy.linalg.orth(weights['input'])
            weights['input'] = weights['input'].reshape(self.filter_shape)
            if self.bias:
                weights["bias"] = scipy.linalg.orth(weights["bias"].reshape((-1, 1))).reshape((-1))
        return weights

    def fit(self, X, y=None):
        self.weights = self.train_algorithm(X)
        self._generate_conv()
        return self.predict(X)

    def update(self, X, y=None, decay=1):
        return self.predict(X)


class ELMConvNaive(ELMConvLayer):
    def __init__(self, spatial=(3, 3), stride=(1, 1),
                 pad=(1, 1)):
        """
        The ELMConvNaive represents a convolutional regression layer within
        an ELM without any constrain. This implementation is based on theano.

        Attributes:
            spatial_extent (optional[int/tuple[int]]): The spatial extent of
                the local receptive field in pixels.
                Size should be (width, height) or an integer, so that the width
                and height are the same. Default is a 3x3 lrf.
            stride (optional[int/tuple[int]]):The stride of the local receptive
                fields in pixels. Size should be (width, height) or an integer,
                so that the width and height are the same. Default is a stride
                of 1x1.
            pad (optional[int/tuple[int]]): The zero-padding of the input layer
                in pixels. Size should be (width, height) or an integer,
                so that the width and height are the same. Default is a padding
                of 1x1.
        """
        super().__init__(1, spatial, stride, pad, 'linear', False)
        self.hidden_matrices = {'K': np.empty((0,0,0,0)),
                                'A': np.empty((0,0,0,0))}
        self.output_gen = None

    def _calc_weights(self):
        K = self.hidden_matrices['K']
        A = self.hidden_matrices['A']
        for i in range(A.shape[1]):
            A[:,i,:,:] = A[:,i,:,:].dot(1/K[i])
        self.weights['input'] = A

    def _generate_conv(self, image_shape=None):
        input = T.tensor4(name='input')
        W = theano.shared(np.asarray(self.weights['input'], dtype=input.dtype),
                          name='W')
        output = T.nnet.conv2d(input, W,
                                 border_mode=self.pad,
                                 subsample=self.stride,
                                 filter_shape=self.hidden_matrices['A'].shape,
                                 input_shape=image_shape)
        self.conv_fct = theano.function([input], output)

    def fit(self, X, y=None):
        self.n_features = y.shape[0]
        if self.output_gen is None:
            input = T.tensor4(name='input')
            target = T.tensor4(name='target')
            output = T.nnet.conv2d(input, target,
                                   border_mode=self.pad,
                                   subsample=self.stride)
            self.output_gen = theano.function([input, target], output)
        self.hidden_matrices['A'] = self.output_gen(y.transpose(1,0,2,3),
                                                    X.transpose(1,0,2,3))
        self.hidden_matrices['K'] = np.sum(np.power(X, 2), axis=(0,2,3))
        self._calc_weights()
        self._generate_conv()
        return self.predict(X)

    def update(self, X, y=None, decay=1):
        self.hidden_matrices['A'] = self.hidden_matrices['A']*decay+\
            self.output_gen(y.transpose(1,0,2,3), X.transpose(1,0,2,3))
        self.hidden_matrices['K'] = self.hidden_matrices['K']*decay+\
                                    np.sum(np.power(X, 2), axis=(0,2,3))
        self._calc_weights()
        self._generate_conv()
        return self.predict(X)

class ELMConvNaive_T(ELMConvLayer):
    def __init__(self, spatial=(3, 3), stride=(1, 1),
                 pad=(1, 1), C=0):
        """
        The ELMConvNaive represents a convolutional regression layer within
        an ELM without any constrain. This implementation is based on theano.

        Attributes:
            spatial_extent (optional[int/tuple[int]]): The spatial extent of
                the local receptive field in pixels.
                Size should be (width, height) or an integer, so that the width
                and height are the same. Default is a 3x3 lrf.
            stride (optional[int/tuple[int]]):The stride of the local receptive
                fields in pixels. Size should be (width, height) or an integer,
                so that the width and height are the same. Default is a stride
                of 1x1.
            pad (optional[int/tuple[int]]): The zero-padding of the input layer
                in pixels. Size should be (width, height) or an integer,
                so that the width and height are the same. Default is a padding
                of 1x1.
        """
        super().__init__(1, spatial, stride, pad, 'linear', False)
        self.hidden_matrices = {'K': None, 'A': None}
        self._C = C
        self.conv_fct = {'train': None, 'predict': None}

    def fit(self, X, y=None):
        self.n_features = y.shape[0]
        self.weights['input'] = theano.shared(value=np.zeros((
            self.n_features, X.shape[1], self.spatial[0], self.spatial[1]),
            dtype=theano.config.floatX), name='w', borrow=True)
        input = T.tensor4(name='input')
        target = T.tensor4(name='target')
        decay = T.scalar(name='decay')
        xy = T.nnet.conv2d(input.transpose(1,0,2,3), target.transpose(1,0,2,3),
                           border_mode=self.pad, subsample=self.stride)
        xx = T.sum(T.power(input, 2), axis=(0,2,3))
        k = ifelse(self.hidden_matrices['input'] is None, )

        lam = theano.shared(value=self._C, name='constrain', borrow=True)
        prediction = T.nnet.conv2d(input, self.weights['input'],
                                   border_mode=self.pad,
                                   subsample=self.stride)
        weights, _ = theano.scan(
            fn=lambda a, k, c: a/(k+c), outputs_info=None,
            sequences=[self.hidden_matrices['A'].transpose(1,0,2,3),
                       self.hidden_matrices['K']], non_sequences=lam)
        new_weights = weights.transpose(1,0,2,3)
        updates = [(self.hidden_matrices['K'],
                    self.hidden_matrices['K'].dot(decay)+xx),
                   (self.hidden_matrices['A'],
                    self.hidden_matrices['A'].dot(decay) + xy),
                   (self.weights['input'], new_weights)]
        self.conv_fct['train'] = theano.function([input, target, decay],
                                                 prediction,
                                                 updates=updates)
        self.conv_fct['predict'] = theano.function([input], prediction)
        return self.conv_fct['train'](X, y, 1)

    def update(self, X, y=None, decay=1):
        return self.conv_fct['train'](X, y, decay)

    def predict(self, X):
        return self.conv_fct['predict'](X)


class ELMConvRidge(ELMConvNaive):
    def __init__(self, spatial=(3, 3), stride=(1, 1),
                 pad=(1, 1), C=2E5):
        """
        The ELMConvRidge represents a convolutional regression layer within
        an ELM with a L2-constrain. This implementation is based on theano.

        Attributes:
            spatial_extent (optional[int/tuple[int]]): The spatial extent of
                the local receptive field in pixels.
                Size should be (width, height) or an integer, so that the width
                and height are the same. Default is a 3x3 lrf.
            stride (optional[int/tuple[int]]):The stride of the local receptive
                fields in pixels. Size should be (width, height) or an integer,
                so that the width and height are the same. Default is a stride
                of 1x1.
            pad (optional[int/tuple[int]]): The zero-padding of the input layer
                in pixels. Size should be (width, height) or an integer,
                so that the width and height are the same. Default is a padding
                of 1x1.
            C (optional[float]): The L2-constrain factor. Default is 2E5.
        """
        super().__init__(spatial, stride, pad)
        self._C = C

    def _calc_weights(self):
        K = self.hidden_matrices['K']
        A = self.hidden_matrices['A']
        for i in range(A.shape[1]):
            A[:, i, :, :] = A[:, i, :, :].dot(1/(1/self._C+K[i]))
        self.weights['input'] = A


class ELMConvAE(ELMConvLayer):
    def __init__(self, n_features, spatial=(3,3), stride=(1,1), pad=(1,1),
                 activation=['sigmoid', 'linear'], bias=True, C=0, ortho=True,
                 rng=None):
        """
        The ELMLRF represents a local receptive field layer within an ELM [1].
        This implementation of ELMLRF is based on theano.

        Attributes:
            n_features (int): Number of features within the layer.
            spatial_extent (optional[int/tuple[int]]): The spatial extent of
                the local receptive field in pixels.
                Size should be (width, height) or an integer, so that the width
                and height are the same. Default is a 3x3 lrf.
            stride (optional[int/tuple[int]]):The stride of the local receptive
                fields in pixels. Size should be (width, height) or an integer,
                so that the width and height are the same. Default is a stride
                of 1x1.
            pad (optional[int/tuple[int]]): The zero-padding of the input layer
                in pixels. Size should be (width, height) or an integer,
                so that the width and height are the same. Default is a padding
                of 1x1.
            activation_funct (str or convolutional activation function):
                The function with which the values should be activated,
                Default is None, because in some layers there is no activation.
            Bias (optional[bool]: If the AE should contain a bias.
                Default is false.
            C (optional[float]): The L2-constrain factor. Default is 0.
            ortho (optional[bool]): If the random weights should be
                orthogonalized or not. Default is False.
            rng (optional[numpy RandomState]): A numpy random state to generate
                the random weights. If no state is given, the seed will be set
                to 42.
        """
        try:
            super().__init__(n_features, spatial, stride, pad, activation[1], bias)
            self.layers = [ELMLRF(n_features, spatial, stride, pad, activation[0],
                                  bias, ortho, rng)]
        except:
            super().__init__(n_features, spatial, stride, pad, activation, bias)
            self.layers = [ELMLRF(n_features, spatial, stride, pad, activation,
                                  bias, ortho, rng)]
        self._C = C
        if self._C>0:
            self.layers.append(ELMConvRidge(C))
        else:
            self.layers.append(ELMConvNaive())

    def _generate_conv(self, image_shape=None):
        input = T.tensor4(name='input')
        W = theano.shared(np.asarray(self.weights['input'], dtype=input.dtype),
                          name='W')
        conv_out = T.nnet.conv2d(input, W,
                               border_mode=self.pad,
                               subsample=self.stride,
                               filter_shape=self.weights['input'].shape,
                               input_shape=image_shape)
        if self.activation_fct is None:
            output = conv_out
        elif self.activation_fct == "hardlimit":
            output = conv_out>0
        elif self.activation_fct == "hardtanh":
            output = T.switch(conv_out > -1, T.switch(conv_out > 1, 1, conv_out), -1)
        else:
            output = self.activation_fct(conv_out)
        self.conv_fct = theano.function([input], output)

    def _calc_weights(self, X, decay=False):
        layer_input = X
        for layer in self.layers:
            if not isinstance(decay, bool):
                layer_input = layer.update(layer_input, X, decay)
            else:
                layer_input = layer.fit(layer_input, X)
        weights = self.layers[-1].weights
        weights['input'] = weights['input'].transpose(1,0,2,3)
        return weights

    def fit(self, X, y=None):
        self.weights = self._calc_weights(X)
        self._generate_conv()
        return self.predict(X)

    def update(self, X, y=None, decay=1):
        self.weights = self._calc_weights(X, decay)
        self._generate_conv()
        return self.predict(X)


