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
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
import matplotlib.pyplot as plt

# Internal modules


__version__ = "0.1"

def x_trans(x):
    return np.sin(x)

x_dimensions = 1
train_size = 100
test_size = 350
hidden_neurons = 10


x_train_size = (train_size, x_dimensions)
x_test_size = (test_size, x_dimensions)

train_x = np.random.uniform(0, 10, size=x_train_size)
train_y = x_trans(train_x)
train_y += np.random.normal(0, np.abs(x_trans(10)/10), size=(train_size, 1))

train_x_scaler = StandardScaler()
train_x = train_x_scaler.fit_transform(train_x)
train_y_scaler = StandardScaler()
train_y = train_y_scaler.fit_transform(train_y)

test_x = np.random.uniform(0, 10, size=x_test_size)
test_y = x_trans(test_x)

test_x = train_x_scaler.transform(test_x)
#print(np.c_[test_x, np.ones(test_x.shape[0])])

# plot train data and real function
x_range = np.arange(-5, 15, 0.1)
fig = plt.figure()
ax = plt.subplot()
plt.scatter(train_x_scaler.inverse_transform(train_x),
            train_y_scaler.inverse_transform(train_y))
plt.plot(x_range, x_trans(x_range), color="0")


#
# instances = [
#     pyextremelm.ELMSKSupervised(
#                  hidden_neurons,
#                  RidgeCV(alphas=[.1, .25, .5, 1, 2, 5, 10, 100],
#                              fit_intercept=False),
#                  activation_funct="sigmoid",
#                  rand_iter=100),
#     pyextremelm.ELMSKSupervised(
#         hidden_neurons,
#         LinearRegression(fit_intercept=False),
#         activation_funct="sigmoid",
#         rand_iter=100),
#     pyextremelm.ELMRegressor(hidden_neurons, rand_iter=100),
#     LinearRegression(fit_intercept=False)]

instances = [
    pyextremelm.ELMRegressor(hidden_neurons, activation_funct="tanh", rand_iter=100),
    pyextremelm.ELMRegressor(hidden_neurons, rand_iter=100),
    LinearRegression(fit_intercept=False)]

i=1
for instance in instances:
    instance.fit(train_x, train_y)
    # print(instance.output_weights.shape)
    prediction = instance.predict(test_x)
    print(
        np.mean((train_y_scaler.inverse_transform(prediction) - test_y) ** 2),
        np.mean(train_y_scaler.inverse_transform(prediction) - test_y))
    y_range = instance.predict(train_x_scaler.transform(x_range.reshape(-1, 1)))
    plt.plot(x_range, train_y_scaler.inverse_transform(y_range), label="%i"%i)
    i+=1

ax.set_xlim(-5, 15)
ax.set_ylim(-2, 2)
plt.legend()
plt.show()
