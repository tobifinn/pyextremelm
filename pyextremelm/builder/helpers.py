# -*- coding: utf-8 -*-
"""
Created on 19.05.16

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
from .activations import named_activations, unnamed_activations

__version__ = "0.1"


def get_activation(funct="sigmoid"):
    """
    Function to get the activation function
    Args:
        funct (str or function):

    Returns:
        function: The activation function
    """
    if isinstance(funct, str) and funct in named_activations:
        return named_activations[funct]
    elif funct is None:
        return named_activations["linear"]
    elif funct in list(named_activations.values()):
        return funct
    elif funct in unnamed_activations:
        return funct
    else:
        raise ValueError(
            "%s isn't implemented yet or isn't an available function name"
            % funct)


def get_dim(X):
    """
    Get the dimensions of X.
    Args:
        X (numpy array): X is the input array (shape: samples*dimensions).

    Returns:
        dimensions (int): The dimensions of X.
    """
    return X.shape[1] if len(X.shape) > 1 else 1


def add_bias(X, bias):
    """
    Convert the X input to a dict with bias.
    Args:
        X (numpy array): The input array.
        bias (bool): If a bias should be added.

    Returns:
        input_dict (dict[numpy array]): The converted input dict.
    """
    if bias:
        input_dict = {"input": X, "bias": np.ones(X.shape[0])}
    else:
        input_dict = {"input": X, "bias": None}
    return input_dict
