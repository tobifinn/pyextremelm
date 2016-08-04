# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:14:25 2016

@author: tfinn
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import scipy.ndimage

from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator

from pyextremelm.builder import ExtremeLearningMachine
from pyextremelm.builder.layers import ELMAE, ELMSparseAE, ELMOrthoRandom
from pyextremelm.builder.layers.convolution import ELMPool, ELMConvAE_linear, ELMLRF

from pyclamster.clustering.kmeans import KMeans

rng = np.random.RandomState(42)
output_layer = 80

elm = ExtremeLearningMachine()
# elm.add_layer(ELMConvAE_linear(10, (3, 11, 11), zero_padding=5, bias=True,
#                      activation='sigmoid'))
# elm.add_layer(ELMConvAE_linear(20, (3, 3), bias=True, zero_padding='full',
#                      activation='linear', C=1))
elm.add_layer(ELMLRF(100, (3,3,3), zero_padding='full', activation='linear'))
elm.add_layer(ELMPool(pooling='squareroot',
                      spatial_extent=(3, 3), stride=1))
elm.add_layer(ELMConvAE_linear(2, (3,3), zero_padding='full', activation='linear', C=10))
elm.add_layer(ELMPool(pooling='squareroot',
                      spatial_extent=(3, 3), stride=1))
# elm.add_layer(ELMConvAE_linear(80, (3, 3), bias=True, zero_padding='full',
#                      activation='tanh', C=1))
# elm.add_layer(ELMPool(pooling='squareroot',
#                       spatial_extent=(3, 3), stride=1))
# elm.add_layer(ELMConvAE_linear(output_layer, (3, 3), bias=True, zero_padding='full',
#                      activation='tanh', C=1))
# elm.add_layer(ELMPool(pooling='squareroot', spatial_extent=(3, 3),
#                       stride=1))
# elm.add_layer(ELMConvAE_linear(2, (1, 1), bias=True, zero_padding='full',
#                      activation='tanh', C=1))
# elm.add_layer(ELMPool(pooling='squareroot', spatial_extent=(3, 3),
#                       stride=1))


images = None
for j in range(1, 5):
    img = scipy.misc.imread('images/Image_Wkm_Aktuell_{0:d}.jpg'.format(j),
                            flatten=False, mode="RGB")
    data = np.asarray(img, dtype='float64')
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

# print(elm.layers[0].weights['input'][10].shape)
# plt.imshow(elm.layers[0].weights['input'][10].transpose(1,2,0), interpolation='none', cmap='gray')
# plt.show()

# # plot original image and first and second components of output
# w = int(np.sqrt(output_layer+1))
# h = int(np.ceil((output_layer+1)/w))
# for i in range(output_layer):
#     plt.subplot(h, w, i + 1);
#     plt.axis('off');
#     plt.imshow(np.mean(elm.layers[0].weights['input'][i].transpose(1, 2, 0), axis=2),
#                interpolation='none', cmap='gray')
# plt.show()
# for n in range(4):
#     plt.subplot(h, w, 1); plt.axis('off'); plt.imshow(
#         scipy.misc.imread('images/Image_Wkm_Aktuell_{0:d}.jpg'.format(n+1),
#                           flatten=False, mode="RGB")[480:1480, 480:1480, :]/256,
#         interpolation='none', cmap='gray')
#     for i in range(output_layer):
#         plt.subplot(h, w, i+2); plt.axis('off'); plt.imshow(filtered_img[n, i, :, :], interpolation='none', cmap='gray')
#     plt.show()

# Clustering test
print(filtered_img.shape)
filtered_img = filtered_img.transpose(0,2,3,1)
#filtered_img = np.concatenate((filtered_img, images), axis=3)
filtered_img = filtered_img-np.mean(filtered_img, axis=(0,1,2))
filtered_img = filtered_img/(np.std(filtered_img, axis=(0,1,2))+0.000001)
b, w, h, c = filtered_img.shape
filtered_img = filtered_img.reshape(-1, c)
print(filtered_img.shape)


# elmae = ExtremeLearningMachine()
# elmae.add_layer(ELMOrthoRandom(2))
# filtered_img = elmae.fit(filtered_img)['output']

# filtered_img = filtered_img-np.mean(filtered_img, axis=0)
# filtered_img = filtered_img/(np.std(filtered_img, axis=0)+0.000001)

try:
    ax = plt.subplot()
    plt.hist2d(filtered_img[:,0], filtered_img[:,1], bins=100, normed=LogNorm)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    plt.colorbar()
    plt.show()
except:
    plt.hist(filtered_img[:,0], bins=100, normed=LogNorm)
    plt.colorbar()
    plt.show()


kmeans = KMeans(2)
kmeans.fit(filtered_img)
labels = kmeans.labels
labels = labels.labels.reshape((b, w, h))
for n in range(4):
    plt.subplot(1, 2, 1); plt.axis('off'); plt.imshow(
        scipy.misc.imread('images/Image_Wkm_Aktuell_{0:d}.jpg'.format(n+1),
                          flatten=False, mode="RGB")[480:1480, 480:1480, :]/256,
        interpolation='none', cmap='gray')
    plt.subplot(1, 2, 2); plt.axis('off'); plt.imshow(labels[n], interpolation='none')
    plt.show()

