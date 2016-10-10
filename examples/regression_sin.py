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
from time import time

# External modules
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Internal modules
from pyextremelm import ELMRegressor
from pyextremelm.builder import ExtremeLearningMachine
from pyextremelm.builder import layers as ELMLayers

__version__ = "0.1"

def x_trans(x):
    return np.sin(x)

x_dimensions = 1
train_size = 100
test_size = 1000
hidden_neurons = 500

x_train_size = (train_size, x_dimensions)
x_test_size = (test_size, x_dimensions)

# Generate the trainings data
train_x = np.random.uniform(0, 10, size=x_train_size)
train_y = x_trans(train_x)
train_y += np.random.normal(0, 0.3, size=(train_size,1))
train_x_scaler = StandardScaler()
train_x = train_x_scaler.fit_transform(train_x)
train_y_scaler = StandardScaler()
train_y = train_y_scaler.fit_transform(train_y)

# Generate the test data
test_x = np.random.uniform(0, 10, size=x_test_size)
test_y = x_trans(test_x)

test_x = train_x_scaler.transform(test_x)

# plot train data and real function
x_range = np.arange(-5, 15, 0.1)
fig = plt.figure()
ax = plt.subplot()
plt.scatter(train_x_scaler.inverse_transform(train_x),
            train_y_scaler.inverse_transform(train_y))
plt.plot(x_range, x_trans(x_range), color="0")


# initialize the extreme learning machines
# # ELM with constraint
elm = ExtremeLearningMachine()
elm.add_layer(ELMLayers.ELMRandom(hidden_neurons))
elm.add_layer(ELMLayers.ELMRegression(C=2E10))# # # ELM without constraint
elmw = ExtremeLearningMachine()
elmw.add_layer(ELMLayers.ELMRandom(hidden_neurons))
elmw.add_layer(ELMLayers.ELMRegression(C=0))
# # # ELM with constraint and a fourier activation
elmf = ELMRegressor(hidden_neurons, activation="fourier", C=2E10)

instances = {
    'Standard': elm,
    'Without constrain': elmw,
    'Fourier': elmf}

for i in instances:
    t0 = time()

    # fit and predict for each instance
    instances[i].fit(train_x, train_y)
    prediction = instances[i].predict(test_x)

    # calculate the forecast performance
    print(
        i,
        np.mean((train_y_scaler.inverse_transform(prediction) - test_y) ** 2),
        np.mean(train_y_scaler.inverse_transform(prediction) - test_y))

    # plot the forecast
    y_range = instances[i].predict(train_x_scaler.transform(x_range.reshape(-1, 1)))
    plt.plot(x_range, train_y_scaler.inverse_transform(y_range), label="%s"%i)

ax.set_xlim(-5, 15)
ax.set_ylim(-2, 2)
plt.legend()
plt.show()
