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
from abc import ABCMeta, abstractmethod

# External modules
import numpy as np
from sklearn.cross_validation import KFold, LeaveOneOut

# Internal modules
from .activations import _activations

__version__ = "0.1"


class ELMBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, hidden_neurons, activation_funct="sigm", bias=True,
                 constraint=None, k=3):
        """
        ELMBase is the extreme learning machine base object.

        Args:
            hidden_neurons (int or str):
                Number of neurons in the hidden layer
                or a method to get the number of hidden_neurons ("cv"/"loo").
            activation_funct (Optional[str or function]):
                The activation function for the hidden layer.
                Defaults is the sigmoid function.
            bias (Optional[bool]): If the random weights should have a bias.
                Defaults is True.
            constraint (Optional[str or None]):
                Method to determine the constraint parameter ("cv"/"loo").
                If None, the output weights aren't constraint.
                Defaults is None.
            k (Optional[int]): Number of folds for the cross-validation.
                Defaults is 3.

        Attributes:
            n_hidden_neurons (int or method):
                Number of neurons in the hidden layer.
            n_input_neurons (int): Number of neurons in the input layer.
            n_output_neurons (int): Number of neurons in the output layer.
            constraint_param (float):
                The constraining parameter.
            k (int): Number of folds for the cross-validation.
                Defaults is 3.
            bias (bool): If the random weights have a bias.
            random_weights (np.array): Connection weights between the input
                and the hidden layer. Size is (number of input neurons + 1) *
                number of hidden neurons.
            output_weights (np.array): Connection weights between the hidden
                and the output layer.
                Size is number of hidden neurons * number of output neurons.
            training_data (Dict[np.array]):
                The training data container (X and (y)).
            activation_funct (function):
                The activation function for the hidden layer.
            constraint_method (str or None):
                Method to determine the constraint parameter ("cv"/"loo").
                If None, the output weights aren't constraint.
            hidden_method (str):
                The method to determine the number of hidden neurons.
                "cv"/"loo"/"fixed"

        """
        self.bias = bias
        self.n_input_neurons = 0
        self.n_output_neurons = 0
        self.constraint_param = 0
        self.k = k
        self.random_weights = np.empty((0, 0))
        self.output_weights = np.empty((0, 0))
        self.training_data = {"X": None, "y": None}
        self.activation_function = self._select_activation(activation_funct)
        self.constraint_method = constraint
        # Get hidden neurons information.
        if isinstance(hidden_neurons, str):
            self.hidden_method = hidden_neurons
            self.n_hidden_neurons = 0
        else:
            assert not isinstance(hidden_neurons, int), \
                "{0} have to be a fixed integer {1} or a method to " \
                "determine the {1} of {0}".format("hidden_neurons", "number")
            self.hidden_method = "fixed"
            self.n_hidden_neurons = hidden_neurons
            #        self.random_weights = np.empty((self.n_input_neurons+int(bias), n_hidden_neurons))
            #        self.output_weights = np.empty((n_hidden_neurons, self.n_output_neurons))

    def _initialize_random_weights(self, n_hidden_neurons=None):
        """
        Initialize the random weights.

        Args:
            n_hidden_neurons (int): Number of hidden neurons.
                If None, self.n_hidden_neurons. Defaults is None.
        """
        if n_hidden_neurons is None:
            n_hidden_neurons = self.n_hidden_neurons
        self.random_weights = np.random.randn(
            self.n_input_neurons + int(self.bias), n_hidden_neurons)

    @staticmethod
    def _get_activation(funct="sigm"):
        """
        Method to get the activation function
        Args:
            funct (str or function):

        Returns:
            function: The activation function
        """
        if isinstance(funct, str):
            assert funct in _activations, \
                "%s isn't implemented yet or isn't an available function name" \
                % (funct)
            return _activations[funct]
        else:
            return funct

    def _get_cv(self, length_array):
        """
        Uses the scikit-learn KFold method for the output.

        Args:
            length_array (int): The length of the array.

        Returns:
            kf (Kfold): The KFold object, iterable.
        """
        kf = KFold(length_array, n_folds=self.k)
        return kf

    @staticmethod
    def _get_loo(length_array):
        """
        Uses the scikit-learn LeaveOneOut method for the output.

        Args:
            length_array (int): The length of the array.

        Returns:
            loo (LeaveOneOut): The LeaveOneOut object, iterable.
        """
        loo = LeaveOneOut(length_array)
        return loo

    @abstractmethod
    def fit(self):
        """
        The public method to fit the extreme learning machine.
        """
        pass

    @abstractmethod
    def _train_fixed(self):
        """
        Method to train the extreme learning machine with cross-validation.
        """
        pass

    @abstractmethod
    def _train(self):
        """
        The private method to train the extreme learning machine
        """
        pass

    @abstractmethod
    def predict(self):
        """
        The method to predict with the trained extreme learning machine.
        """
        pass


if __name__ == "__main__":
    ELMBase(5)