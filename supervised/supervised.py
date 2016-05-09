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

# Internal modules
from ..base import ELMBase

__version__ = "0.1"


class ELMSupervised(ELMBase):
    def fit(self, X, y):
        """
        The supervised training method.

        Args:
            X (np.array): The training input array.
            y (np.array): The training output array.
        """
        #
        if self.bias:
            X = np.column_stack([X, np.ones([X.shape[0], 1])])
        #self.training_data = {"X": X, "y": y}
        if self.hidden_method in ["cv", "loo"]:
            self._train_cv(X, y)
        else:   # fixed number
            self._train_fixed(X, y)

    def _train(self, X, y):
        pass


class ELMSKSupervised(ELMBase):
    pass
