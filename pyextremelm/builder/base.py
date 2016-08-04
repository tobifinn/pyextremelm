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
import time


# External modules

# Internal modules

class ExtremeLearningMachine(object):
    """
    This class is a container for the layers for a neural network based on the
    extreme learning machine [1].
    ---------------------------------------------------------------------------
    [1] Huang, G. B., Zhu, Q. Y., & Siew, C. K. (2006).
        Extreme learning machine: theory and applications. Neurocomputing,
        70(1), 489-501.
    """

    def __init__(self):
        self.layers = []
        self._timer = None

    def _set_timer(self):
        self._timer = time.time()

    def add_layer(self, layer):
        """
        This method adds a pre-configured layer to the layer list.
        Args:
            layer (Instance of child of ELMLayer): The pre-configured layer.

        Returns:
            self: returns an instance of self.
        """
        self.layers.append(layer)
        return self

    def fit(self, X, y=None):
        """
        This method trains the nn for supervised and unsupervised training.
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
        self._set_timer()
        training_info["samples"] = X.shape[0]
        output = X
        for layer in self.layers:
            input = output
            output = layer.fit(input, y)
        training_info['output'] = output
        training_info['train_time'] = time.time() - self._timer
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

    def fit_batch(self, X, y=None):
        """
        Method to fit the neural network in a batchwise manner.
        Args:
            X (numpy array): The input data field.
            y (optional[numpy array]): The output data field.
                In unsupervised learning this is only for consistency.

        Returns:

        """
        pass

    def update(self, X, y=None, decay=1):
        """
        Method to update the neural network.
        Args:
            X (numpy array): The input data field.
            y (optional[numpy array]): The output data field.
                In unsupervised learning this is only for consistency.
            decay_factor (optional[float]): The decay factor for the update.
                It can be seen like the learning rate of the update. Values
                greater than 1 are leading to a decay of older fits. So this
                factor could be used, to update the NN wth in a time decaying
                system. Default is 1, so every update step is weighted equally,
                without any decaying.
        Returns:
            update_info (dict): A dict with update information.
                samples (integer): How many samples are used for the update.
                loss (float): An error approximation for the training dataset,
                    for the given loss function of the network.
                train_time (float): The update duration for the whole
                    network in seconds.
                training_output (numpy array): The forecasted output for the
                    update dataset. Could be used for further network steps.
        """
        update_info = {"samples": None, "loss": None, "train_time": None,
                         "output": None}
        self._set_timer()
        update_info["samples"] = X.shape[0]
        output = X
        for layer in self.layers:
            input = output
            output = layer.update(input, y, decay)
        update_info['output'] = output
        update_info['train_time'] = time.time() - self._timer
        return update_info

    def print_network_structure(self):
        """
        Print the network structure as formatted string with information about
        the different layers.
        Returns:
            structure (str): The formatted network structure.
        """
        s = [str(l) for l in self.layers]
        s = '\n'.join(s)
        return '\033[1m' + 'Network structure\n' + '\033[0m' + s


class ELMClustering(object):
    """
    This class is a container for layers of the extreme learning machine, so
    that the extreme learning machine could be used to cluster the data with a
    method proposed by Duan et al., 2016 [1]. With this method the data is
    clustered in a manner like the k-means algorithm. The idea is that the ELM
    classification algorithm could be used to classify the data in an
    unsupervised way.
    ---------------------------------------------------------------------------
    [1] Duan, L., Yuan, B., Cui, S., Miao, J., & Zhu, W. (2016).
        KELMC: An Improved K-Means Clustering Method Using Extreme Learning
        Machine. In Proceedings of ELM-2015 Volume 2 (pp. 273-283). Springer
        International Publishing.
    """
    pass
