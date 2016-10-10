# -*- coding: utf-8 -*-
"""
Created on 02.08.16

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
from copy import deepcopy

# External modules
import numpy as np
import scipy.linalg
import scipy.sparse.linalg

# Internal modules
from .base import ELMLayer


class ELMRidge(ELMLayer):
    def __init__(self, C=0):
        """
        Extreme learning machine regression layer [1], based on the ridge
        regression. The equations are solved by scipy.linalg.lstsq.
        Args:
            C (optional[float]): The constrain factor for the ridge regression.
                In literature this factor is also called lambda. A large C
                equals a large regularization and vice versa.
                Default value is 0.
        """
        super().__init__(n_features=1, activation='linear', bias=False)
        self._C = C
        self.hidden_matrices = {'K': np.empty((0,0)), 'A': np.empty((0,0))}

    def __str__(self):
        s = super().__str__()[:-1]
        s += ", L2-constrain: {0:s})".format(str(self._C))
        return s

    def _calc_weights(self):
        K = self.hidden_matrices['K']
        A = self.hidden_matrices['A']
        if self._C>0:
            self.weights['input'] = scipy.linalg.lstsq(
                (K+np.ones(K.shape[0]).dot(self._C)), A)[0]
        else:
            self.weights['input'] = scipy.linalg.lstsq(K, A)[0]            
        self.activation_fct.weights = self.weights

    def fit(self, X, y=None):
        """
        Method to fit the regression.
        Args:
            X (numpy array): The input data array.
            y (numpy array): The target data array.

        Returns:
            fitted_X (numpy array): The output data array of the layer.
        """
        self.n_features=y.shape[1] if y.ndim > 1 else 1
        self.hidden_matrices['K'] = X.T.dot(X)
        self.hidden_matrices['A'] = X.T.dot(y)
        self._calc_weights()
        return self.predict(X)

    def update(self, X, y=None, decay=1):
        """
        Method to update the layer weights with given arrays. This update
        function is based on the recursive least square function, such that
        only the sample covariance and cross-covariance needs only to be
        updated. These updated matrices are fed into a linear equation solver.
        This procedure was introduced by McDonnel et al. [1] into the extreme
        learning machine community.
        Args:
            X (numpy array): The input data array.
            y (numpy array): The target data array.
            decay (optional[float]): The decay factor determines the decaying
                of old partial weights. It is also possible to weight the
                update with a varying decay factor.

        Returns:
            fitted_X (numpy array): The output data array of the layer.
        -----------------------------------------------------------------------
        [1] McDonnell, M. D., Tissera, M. D. et al. (2015).
            Fast, simple and accurate handwritten digit classification by
            training shallow neural network classifiers with the
            ‘extreme learning machine’algorithm. PloS one, 10(8)
        """
        self.hidden_matrices['K'] = self.hidden_matrices['K'].dot(decay) + X.T.dot(X)
        self.hidden_matrices['A'] = self.hidden_matrices['A'].dot(decay) + X.T.dot(y)
        self._calc_weights()
        return self.predict(X)


class ELMSkReg(ELMRidge):
    """
    A regression
    """
    def __init__(self, algo):
        """
        Extreme learning machine regression layer, based on a pre-configured
        scikit-learn regression. The fitting and prediction method are given by
        the scikit-learn algorithm.
        Args:
            algo (scikit learn algorithm): The preconfigured scikit-learn
                algorithm.
                E.g. sklearn.linear_model.Lasso(alpha=0.5, fit_intercept=False)
        """
        super().__init__()
        self.algo = algo
        self.hidden_matrices = {'K': np.empty((0, 0)),
                                'A': np.empty((0, 0))}

    def __str__(self):
        s = super().__str__()[:-1]
        s += ", {0:s})".format(str(self.algo))
        return s

    def _calc_weights(self):
        K = self.hidden_matrices['K']
        A = self.hidden_matrices['A']
        self.algo.fit(K, A)

    def predict(self, X):
        """
        Method to make a prediction with the layer.
        Args:
            X (numpy array): The input data array for the layer.

        Returns:
            prediction (numpy array): The predicted data array of the layer.
        """
        return self.algo.predict(X)


class ELMRegression(ELMRidge):
    """
    An alias class for the main regression method (at the moment ELMRidge).
    """
    pass
