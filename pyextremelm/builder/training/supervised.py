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

from sklearn.linear_model import Lasso

# Internal modules

__version__ = "0.1"


def ELMNaive(X=None, y=None, n_neurons=1, bias=False):
    return {"input": np.linalg.pinv(X).dot(y), "bias": None}

def ELMRidge(X=None, y=None, n_neurons=1, bias=False, C=0):
    if C>0:
        factors = np.linalg.inv(X.T.dot(X)+np.eye(X.shape[1])/C).dot(X.T).dot(y)
    else:
        factors = np.linalg.pinv(X).dot(y)
    return {"input": factors, "bias": None}

def ELMLasso(X=None, y=None, n_neurons=1, bias=False, C=0):
    if C>0:
        factors = Lasso(alpha=1/C, fit_intercept=False).fit(X, y).coef_
    else:
        factors = np.linalg.pinv(X).dot(y)
    return {"input": factors, "bias": None}
