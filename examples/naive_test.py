# -*- coding: utf-8 -*-
"""
Created on 10.05.16
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
import pyextremelm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV

# Internal modules


__version__ = "0.1"

train_size = 10000
test_size = 1000

hidden_neurons = 10

train_x = np.random.uniform(size=(train_size,4))
train_y = train_x[:,0]*0.5 + train_x[:,1]**(1/2)*0.3 + train_x[:,2]**(5)*0.8 \
          + np.random.normal(0, 1, size=train_size)


train_x_scaler = StandardScaler()
train_x = train_x_scaler.fit_transform(train_x)
train_y_scaler = StandardScaler()
train_y = train_y_scaler.fit_transform(train_y)

test_x = np.random.uniform(size=(test_size,4))
test_y = test_x[:,0]*0.5 + test_x[:,1]**(1/2)*0.3 + test_x[:,2]**(5)*0.8

test_x = train_x_scaler.transform(test_x)

instances = [pyextremelm.ELMRegressor(hidden_neurons, rand_iter=100),
             pyextremelm.ELMSKSupervised(
                 hidden_neurons,
                 RidgeCV(alphas=[.1, .25, .5, 1, 2, 5, 10, 100],
                             fit_intercept=False),
                 rand_iter=100),
             LinearRegression(fit_intercept=False),
             ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                          fit_intercept=False),
             RidgeCV(alphas=[.1, .25, .5, 1, 2, 5, 10, 100],
                     fit_intercept=False)]

for instance in instances:
    instance.fit(train_x, train_y)
    prediction = instance.predict(test_x)
    print(
        np.mean((train_y_scaler.inverse_transform(prediction) - test_y) ** 2))
