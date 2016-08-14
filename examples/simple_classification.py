# -*- coding: utf-8 -*-
"""
Created on 08.06.16

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
from time import time

# External modules
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss

# Internal modules
from pyextremelm import ELMClassifier
from pyextremelm.builder import metrics
from pyextremelm.builder import ExtremeLearningMachine
from pyextremelm.builder import layers as ELMLayers


def x_trans(x):
    return (np.sin(np.sum(x, axis=1))>0).astype(int)

x_dimensions = 5
train_size = 10000
test_size = 1000
hidden_neurons = 50

classifier = ELMLayers.ELMSoftMax()


x_train_size = (train_size, x_dimensions)
x_test_size = (test_size, x_dimensions)


train_x = np.random.normal(0, 1, size=x_train_size)
train_y = x_trans(train_x+np.random.normal(0, 0.1, size=x_train_size))
train_y = classifier.labels_bin(train_y)
train_x_scaler = StandardScaler()
train_x = train_x_scaler.fit_transform(train_x)

test_x = np.random.normal(0, 1, size=x_test_size)
test_y = x_trans(test_x)
test_x = train_x_scaler.transform(test_x)


elm = ELMClassifier(hidden_neurons, C=2E5)

elmae = ExtremeLearningMachine()
elmae.add_layer(ELMLayers.ELMAE(hidden_neurons, C=0))
elmae.add_layer(ELMLayers.ELMAE(hidden_neurons, C=0))
elmae.add_layer(ELMLayers.ELMNaive())
elmae.add_layer(classifier)


instances = [
    elm,
    elmae]



i=1
for instance in instances:
    t0 = time()
    # fit and predict for each instance
    instance.fit(train_x, train_y)
    prediction, prob, _ = instance.predict(test_x)
    # calculate the forecast performance
    print(instance.print_network_structure())
    print('Brier score: {0:f}'.format(
        brier_score_loss(elm.labels_bin(test_y)[:,1], prob[:,1])))
    print('Time elapsed: {0:f}'.format(time()-t0))
    print(test_y[:20], '\n', prediction[:20])
