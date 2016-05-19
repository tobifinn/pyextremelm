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

# Internal modules


__version__ = "0.1"


class ExtremeLearningMachine(object):
    """
    This class is a container for the layers of the extreme learning machine.

    Attributes:
        layers (List[ELMLayer]):
            The list with the added layers for the extreme learning machine.
    """
    def __init__(self):
        self.layers = []

    def add_existing_layer(self, layer):
        """
        This method adds a pre-configured layer to the layer list.
        Args:
            layer (ELMLayer): The pre-configured layer.

        Returns:
            self: returns an instance of self.
        """
        assert isinstance(layer, ELMLayer)
        self.layers.append(layer)
        return self

    def add_new_layer(self, neurons, train_method, activation="sigmoid"):
        """
        This method creates a new layer and adds it to the layer list.

        Args:
            neurons (int): Number of neurons in the layer.
            train_method (ELMTraining): The training method of the layer.
            activation (optional[str or numpy function]):
                The activation function for the layer.

        Returns:
            self: returns an instance of self.
        """
        layer = ELMLayer(neurons, train_method, activation)
        self.add_existing_layer(layer)
        return self

    def fit(self, X, y=None):
        """
        This method trains the elm for supervised and unsupervised training.
        Args:
            X (numpy array): The input data field.
            y (optional[numpy array]): The output data field.
                In unsupervised learning this is only for consistency.

        Returns:
            self: returns an instance of self.
        """
        output = X
        for layer in self.layers:
            input=output
            output = layer.train(input, y)
        return self

    def predict(self, X):
        """
        This method predict the outcome for given input x.
        Args:
            X (numpy array): The input data field.

        Returns:
            prediction (numpy array): The predicted data field.
        """
        prediction = X
        for layer in self.layers:
            input = prediction
            prediction = layer.predict(input)
        return prediction


class ELMLayer(object):
    pass