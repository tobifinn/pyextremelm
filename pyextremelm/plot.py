# -*- coding: utf-8 -*-
"""
Created on 12.05.16
Created for pyextremelm

Based on: github.com/miloharper/visualise-neural-network

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
import matplotlib.pyplot as plt
from math import atan, sin, cos

# Internal modules


__version__ = "0.1"


class Neuron(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, fill=False):
        circle = plt.Circle((self.x, self.y), radius=1, fill=fill)
        plt.gca().add_patch(circle)

class Dots(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, fill=False):
        y=self.y+0.5
        for i in range(3):
            circle = plt.Circle((self.x, y), radius=0.1, fill="0")
            plt.gca().add_patch(circle)
            y += 0.4


class Layer(object):
    def __init__(self, network, number_of_neurons):
        self.previous_layer = self.__get_previous_layer(network)
        self.x = self.__calculate_layer_x_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __calculate_layer_x_position(self):
        if self.previous_layer:
            return self.previous_layer.x + 6
        else:
            return 0

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        # create neurons
        if number_of_neurons<6:
            y = 4 * (6 - number_of_neurons) / 2
            for iteration in range(number_of_neurons):
                neuron = Neuron(self.x, y)
                neurons.append(neuron)
                y += 4
        # Plot dots instead of too much neurons
        else:
            y=2
            neuron = Neuron(self.x, y)
            neurons.append(neuron)
            y += 2
            neuron = Dots(self.x, y)
            neurons.append(neuron)
            y += 4
            for iteration in range(2):
                neuron = Neuron(self.x, y)
                neurons.append(neuron)
                y += 4
        return neurons

    def __line(self, neuron1, neuron2):
        line = plt.Line2D(
            # (neuron1.x - x_adjustment, neuron2.x + x_adjustment),
            # (neuron1.y - y_adjustment, neuron2.y + y_adjustment))
            (neuron1.x-1, neuron2.x+1),
            (neuron1.y, neuron2.y))
        plt.gca().add_line(line)

    def draw(self):
        for neuron in self.neurons:
            neuron.draw()
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    if not isinstance(neuron, Dots)\
                            and not isinstance(previous_layer_neuron, Dots):
                        self.__line(neuron, previous_layer_neuron)


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, number_of_neurons):
        layer = Layer(self, number_of_neurons)
        self.layers.append(layer)

    def draw(self):
        for layer in self.layers:
            layer.draw()
        plt.axis('scaled')
        plt.show()

if __name__ == "__main__":
    network = NeuralNetwork()
    network.add_layer(3)
    network.add_layer(8)
    network.add_layer(6)
    network.draw()