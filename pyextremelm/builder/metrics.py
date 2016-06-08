# -*- coding: utf-8 -*-
"""
Created on 20.05.16

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

# External modules
import numpy as np

# Internal modules


__version__ = ""


class Metric(object):
    def __init__(self, prediction, true=None):
        self.prediction = prediction
        self.true = true

    @abc.abstractmethod
    def calc_score(self):
        pass

    @abc.abstractmethod
    def better(self, value):
        pass

    @property
    def score(self):
        return self.calc_score()


class MeanSquaredError(Metric):
    def calc_score(self):
        return np.mean((self.prediction-self.true)**2)

    def better(self, value):
        if self.score < value:
            return True
        else:
            return False

class NoMetric(Metric):
    def calc_score(self):
        return 0

    def better(self, value):
        return True