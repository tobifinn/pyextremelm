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
import scipy.misc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Internal modules
import pyextremelm.builder.metrics as metrics
import pyextremelm.builder as ELM
import pyextremelm.builder.layers as ELMLayers

__version__ = ""

n_neurons = 10
image_center = int(1920 / 2)


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


original_image = scipy.ndimage.imread(u'Image_Wkm_Aktuell_3.jpg', flatten=True,
                                      mode="RGB")
original_image = original_image - scipy.ndimage.uniform_filter(
    original_image, size=(100, 100), mode="constant")
original_image = original_image[480:1480, 480:1480]

samples, size = split_image(original_image, 10)

elmae = ELM.ExtremeLearningMachine(metrics.MeanSquaredError, 1)
# elmae.add_existing_layer(ELMLayers.ELMAE(n_images, C=10E10))
elmae.add_existing_layer(ELMLayers.ELMSparseAE(n_neurons, C=1))
elmae.add_existing_layer(ELMLayers.ELMAE(n_neurons, C=10E10))

elmae.fit(samples, None)
# print(elmae.layers[0].weights)
# output = elmae.predict(split_image(original_image, 50, 1)[0])
# print(output.shape)
# output = output.reshape((size[0], size[1], 10))
#print(split_image(original_image, 50, 1)[0].shape)
output = elmae.layers[0].weights["input"].reshape((size[0], size[1],n_neurons))
#output = split_image(original_image, 50, 1)[0].dot(output)
#print(output.shape)

print(np.var(original_image))
fig = plt.figure()
gs = gridspec.GridSpec(4, 5)
ax = plt.subplot(gs[:2, :])
# plt.imshow(original_image)
plt.imshow(original_image)
for i in range(output.shape[2]):
    ax = plt.subplot(gs[int(i / 5) - 2, i % 5])
    plt.imshow(output[:, :, i], interpolation="nearest")
    print(np.var(output[:, :, i]))
plt.tight_layout()
plt.show()
