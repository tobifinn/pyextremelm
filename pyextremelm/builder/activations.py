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
import abc

# External modules
import numpy as np
import scipy.special

# Internal modules


__version__ = "0.1"


class Activation(object):
    def __init__(self, weights):
        self.weights = weights

    @abc.abstractmethod
    def activate(self, X):
        pass


class Sigmoid(Activation):
    """
    The sigmoid activation.
    Args:
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    def activate(self, X):
        if self.weights["bias"] is None:
            weights_array = self.weights["input"]
            X_array = X["input"]
        else:
            weights_array = np.r_[self.weights["input"], self.weights["bias"]]
            X_array = np.c_[X["input"], X["bias"]]
        nonactivated_array = X_array.dot(weights_array)
        array_activated = scipy.special.expit(nonactivated_array)
        return array_activated


class Tanh(Activation):
    """
    The tanh activation.
    Args:
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    def activate(self, X):
        if self.weights["bias"] is None:
            weights_array = self.weights["input"]
            X_array = X["input"]
        else:
            weights_array = np.r_[self.weights["input"], self.weights["bias"]]
            X_array = np.c_[X["input"], X["bias"]]
        nonactivated_array = X_array.dot(weights_array)
        activated_array = np.tanh(nonactivated_array)
        return activated_array


class Fourier(Activation):
    """
    The fourier activation.
    Args:
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    def activate(self, X):
        if self.weights["bias"] is None:
            weights_array = self.weights["input"]
            X_array = X["input"]
        else:
            weights_array = np.r_[self.weights["input"], self.weights["bias"]]
            X_array = np.c_[X["input"], X["bias"]]
        nonactivated_array = X_array.dot(weights_array)
        activated_array = np.cos(nonactivated_array)
        return activated_array


class Gaussian(Activation):
    """
    The gaussian activation.
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    def activate(self, X):
        return np.exp(-self.weights["bias"] * np.abs(
            X["input"] - self.weights["input"]))


class Multiquadratic(Activation):
    """
    The multiquadratic activation.
    Args:
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    def activate(self, X):
        return np.sqrt(np.power(self.weights["bias"], 2) +
                       np.abs(X["input"] - self.weights["input"]))


class Hardlimit(Activation):
    """
    The hardlimit activation.
    Args:
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    def activate(self, X):
        return ((X["input"].dot(self.weights["input"]) +
                 0 if self.weights["bias"] is None else self.weights["bias"])
                <= 0).astype(int)


class Linear(Activation):
    """
    The linear activation, which combines the weights with the input.
    Args:
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    def activate(self, X):
        if self.weights["bias"] is None:
            weights_array = self.weights["input"]
            X_array = X["input"]
        else:
            weights_array = np.r_[self.weights["input"], self.weights["bias"]]
            X_array = np.c_[X["input"], X["bias"]]
        activated_array = X_array.dot(weights_array)
        return activated_array


class Relu(Activation):
    """
    The relu activation, which combines the weights with the input.
    Args:
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    def activate(self, X):
        if self.weights["bias"] is None:
            weights_array = self.weights["input"]
            X_array = X["input"]
        else:
            weights_array = np.r_[self.weights["input"], self.weights["bias"]]
            X_array = np.c_[X["input"], X["bias"]]
        activated_array = X_array.dot(weights_array)
        activated_array[activated_array<0] = 0
        return activated_array

named_activations = {"sigmoid": Sigmoid, "tanh": Tanh, "linear": Linear,
                     "gaussian": Gaussian, "multiquadratic": Multiquadratic,
                     "hardlimit": Hardlimit, "fourier": Fourier, "relu": Relu}
unnamed_activations = []
