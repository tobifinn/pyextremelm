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
import abc
from copy import deepcopy
from time import time

# External modules
import numpy as np

# Internal modules


class ExtremeLearningMachine(object):
    """
    This class is a container for the layers of the extreme learning machine.

    Args:
        layers (optional[List[ELMLayer]]):
            The list with the added layers for the extreme learning machine.
            The default is an empty list, so no layer is added to the container.
    """
    def __init__(self, layers=None):
        self.layers = layers
        if self.layers is None:
            self.layers = []
        self.train_fct = None
        self.predict_fct = None
        self.functions = [self.train_fct, self.predict_fct]

    def add_layer(self, layer):
        """
        This method adds a pre-configured layer to the layer list.
        Args:
            layer (Instance of child of ELMLayer): The pre-configured layer.

        Returns:
            self: returns an instance of self.
        """
        try:
            layer.update
        except Exception as e:
            raise ValueError('The given layer isn\'t an available elm layer')
        self.layers.append(layer)
        return self

    def fit(self, X, y=None):
        """
        This method trains the elm for supervised and unsupervised training.
        Args:
            X (numpy array): The input data field.
            y (optional[numpy array]): The output data field.
                In unsupervised learning this is only for consistency.

        Returns:
            trainings_info (dict): A dict with trainings informations.
                samples (integer): How many samples are used to train.
                loss (float): An error approximation for the training dataset,
                    for the given loss function of the network.
                train_time (float): The training duration for the whole
                    network in seconds.
                training_output (numpy array): The forecasted output for the
                    training dataset. Could be used for further network steps.

        """
        training_info = {"samples": None, "loss": None, "train_time": None,
                         "output": None}
        t0 = time()
        training_info["samples"] = X.shape[0]
        output = X
        for layer in self.layers:
            input = output
            output = layer.fit(input, y)
        # self.loss(y, loss)
        training_info['output'] = output
        training_info['train_time'] = time() - t0
        return training_info

    def predict(self, X):
        """
        This method predict the outcome for given input x.
        Args:
            X (numpy array): The input data field.

        Returns:
            prediction (numpy array): The predicted data field.
            prediction_time (float): The prediction duration in seconds.
        """
        t0 = time()
        prediction = X
        for layer in self.layers:
            input = prediction
            prediction = layer.predict(input)
        return prediction

    def update(self, X, y=None):
        pass

    def print_network_structure(self):
        s = [str(l) for l in self.layers]
        s = '\n'.join(s)
        return '\033[1m' + 'Network structure\n' + '\033[0m' + s
