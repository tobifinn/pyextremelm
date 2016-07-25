# -*- coding: utf-8 -*-
"""
Created on 22.05.16

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
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Internal modules
import pyextremelm.builder as ELM
import pyextremelm.builder.layers as ELMLayers

__version__ = "0.1"

image_center = int(1920 / 2)
n_neurons = 20

def split_image(image, parts, part_nr=None):
    width = [int(image.shape[0] / parts), int(image.shape[1] / parts)]
    parts = [range(0, image.shape[0], width[0]),
             range(0, image.shape[1], width[1])]
    samples = None
    if part_nr is None:
        for key, part in enumerate(parts[0]):
            mini_image = image[part:part + width[0],
                         parts[1][key]:parts[1][key] + width[1]]
            w, h = tuple(mini_image.shape)
            mini_image = np.reshape(mini_image, (1, w * h))
            if samples is None:
                samples = mini_image
            else:
                samples = np.r_[samples, mini_image]
    else:
        mini_image = image[
                     parts[0][part_nr]:parts[0][part_nr] + width[0],
                     parts[1][part_nr]:parts[1][part_nr] + width[1]]
        w, h = tuple(mini_image.shape)
        samples = np.reshape(mini_image, (1, w * h))
    return samples, (w, h)

trainig_samples = None
for j in range(1, 5):
    img = scipy.misc.imread(u'images/Image_Wkm_Aktuell_{0:d}.jpg'.format(j),
                            flatten=True, mode="RGB")
    data = np.asarray(img, dtype='float64')/256
    data = data[480:1480, 480:1480]
    data = data - data.mean(axis=(0,1))
    S = data.std(axis=(0,1))
    data = data / (S + 0.02)
    samples, size = split_image(data, 20)
    if trainig_samples is None:
        trainig_samples = samples
    else:
        trainig_samples = np.concatenate((trainig_samples, samples))

elmae = ELM.ExtremeLearningMachine()
elmae.add_layer(ELMLayers.ELMAE(n_neurons, activation="sigmoid", C=0.1))
elmae.fit(trainig_samples, None)


weights = elmae.layers[0].weights["input"].reshape((size[0], size[1],n_neurons))
for i in range(0, weights.shape[2]):
    weights[:,:,i] /= np.sqrt(np.sum(np.power(weights[:,:,i], 2)))
rows = int(np.ceil(weights.shape[2]/5))
fig = plt.figure(0)
fig.suptitle("Maximum activation for the first ae layer")
gs = gridspec.GridSpec(rows, 5)
for i in range(weights.shape[2]):
    ax = plt.subplot(gs[int(i / 5) - rows, i % 5])
    plt.imshow(weights[:, :, i], interpolation="nearest", cmap='gray')
plt.show()
