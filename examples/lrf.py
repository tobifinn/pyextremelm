# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:14:25 2016

@author: tfinn
"""

import numpy as np

import matplotlib.pyplot as plt

import scipy.ndimage
import numpy

from pyextremelm.builder import ExtremeLearningMachine
from pyextremelm.builder.layers.convolution import ELMPool, ELMConvAE_linear

rng = np.random.RandomState(42)

output_layer = 10

elm = ExtremeLearningMachine()
elm.add_layer(ELMConvAE_linear(20, (5, 5), bias=True, C=0.1, activation='relu'))
elm.add_layer(ELMPool(pooling='max'))
elm.add_layer(ELMConvAE_linear(40, (5, 5), bias=True, C=0.1, activation='relu'))
elm.add_layer(ELMPool(pooling='max'))
elm.add_layer(ELMConvAE_linear(60, (5, 5), bias=True, C=0.1, activation='relu'))
elm.add_layer(ELMPool(pooling='max'))
elm.add_layer(ELMConvAE_linear(10, (5, 5), bias=True, C=0.1, activation='relu'))

images = None
for j in range(1, 5):
    img = scipy.misc.imread('images/Image_Wkm_Aktuell_{0:d}.jpg'.format(j),
                            flatten=False, mode="RGB")
    data = numpy.asarray(img, dtype='float64')
    data = data[480:1480, 480:1480, :]/256
    data = scipy.misc.imresize(data, (256, 256), 'nearest')
    data = data - data.mean(axis=(0,1))
    S = data.std(axis=(0,1))
    data = data / (S + 0.02)
    data = data.reshape(1, 256, 256, 3)
    if images is None:
        images = data
    else:
        images = np.concatenate((images, data), axis=0)

img_ = images.transpose(0, 3, 1, 2)

filtered_img = elm.fit(img_)['output']

plt.imshow(elm.layers[0].weights['input'][10].transpose(1,2,0), interpolation='none', cmap='gray')
plt.show()

# plot original image and first and second components of output
w = int(np.sqrt(output_layer+1))
h = int(np.ceil((output_layer+1)/w))
for n in range(4):
    plt.subplot(h, w, 1); plt.axis('off'); plt.imshow(images[n, :, :, :], interpolation='none', cmap='gray')
    for i in range(output_layer):
        plt.subplot(h, w, i+2); plt.axis('off'); plt.imshow(filtered_img[n, i, :, :], interpolation='none', cmap='gray')
    plt.show()
