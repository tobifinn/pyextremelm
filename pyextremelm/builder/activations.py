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
import numpy as np
import scipy.special

# Internal modules


__version__ = "0.1"


def sigmoid(X, weights):
    """
    The sigmoid activation.
    Args:
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    if weights["bias"] is None:
        weights_array = weights["input"]
        X_array = X["input"]
    else:
        weights_array = np.r_[weights["input"], weights["bias"]]
        X_array = np.c_[X["input"], X["bias"]]
    nonactivated_array = X_array.dot(weights_array)
    array_activated = scipy.special.expit(nonactivated_array)
    return array_activated


def tanh(X, weights):
    """
    The tanh activation.
    Args:
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    if weights["bias"] is None:
        weights_array = weights["input"]
        X_array = X["input"]
    else:
        weights_array = np.r_[weights["input"], weights["bias"]]
        X_array = np.c_[X["input"], X["bias"]]
    nonactivated_array = X_array.dot(weights_array)
    activated_array = np.tanh(nonactivated_array)
    return activated_array


def fourier(X, weights):
    """
    The fourier activation.
    Args:
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    if weights["bias"] is None:
        weights_array = weights["input"]
        X_array = X["input"]
    else:
        weights_array = np.r_[weights["input"], weights["bias"]]
        X_array = np.c_[X["input"], X["bias"]]
    nonactivated_array = X_array.dot(weights_array)
    activated_array = np.cos(nonactivated_array)
    return activated_array


def gaussian(X, weights):
    """
    The gaussian activation.
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    return np.exp(-weights["bias"] * np.abs(X["input"] - weights["input"]))


def multiquadratic(X, weights):
    """
    The multiquadratic activation.
    Args:
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    return np.sqrt(np.power(weights["bias"], 2) +
                   np.abs(X["input"] - weights["input"]))


def hardlimit(X, weights):
    """
    The hardlimit activation.
    Args:
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    return ((X["input"].dot(weights["input"]) + weights["bias"]) <= 0).astype(int)


def linear(X, weights):
    """
    The linear activation, which combines the weights with the input.
    Args:
        X (dict[numpy array]): The input dict with input and bias key.
        weights (numpy array): The weights dict with input and bias key.

    Returns:
        activated_array (np.array): The activated array.
    """
    if weights["bias"] is None:
        weights_array = weights["input"]
        X_array = X["input"]
    else:
        weights_array = np.r_[weights["input"], weights["bias"]]
        X_array = np.c_[X["input"], X["bias"]]
    activated_array = X_array.dot(weights_array)
    return activated_array


named_activations = {"sigmoid": sigmoid, "tanh": tanh, "linear": linear,
                     "gaussian": gaussian, "multiquadratic": multiquadratic,
                     "hardlimit": hardlimit, "fourier": fourier}
unnamed_activations = []
