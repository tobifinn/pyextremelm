# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:14:25 2016

@author: tfinn
"""
import os
import glob
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import scipy.ndimage

from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator

from pyextremelm.builder import ExtremeLearningMachine
from pyextremelm.builder.layers.convolution import ELMLRF, ELMConvAE

from pyclamster.clustering.kmeans import KMeans


image_directory = '/home/tfinn/Data/Cloud_camera/wolf/cam3'
all_images = glob.glob(os.path.join(image_directory, "Image_*.jpg"))

rng = np.random.RandomState(1)
output_layer = 19

def image_iterator(batch_size=10):
    for i in range(0, len(all_images), batch_size):
        images = None
        for j in range(batch_size):
            img = scipy.misc.imread(all_images[i+j], flatten=False, mode="RGB")
            data = np.asarray(img, dtype='float64')
            data = data[480:1480, 480:1480, :] / 256
            data = scipy.misc.imresize(data, (256, 256), 'nearest')
            data = data - data.mean(axis=(0, 1))
            S = data.std(axis=(0, 1))
            data = data / (S + 0.000002)
            data = data.reshape(1, 256, 256, 3)
            if images is None:
                images = data
            else:
                images = np.concatenate((images, data), axis=0)
        img_ = images.transpose(0, 3, 1, 2)
        yield img_


elm = ExtremeLearningMachine()
#elm.add_layer(ELMLRF(output_layer, ortho=True, activation='linear', rng=rng))
elm.add_layer(ELMConvAE(output_layer, spatial=(5,5), ortho=True,
                        activation=['sigmoid', 'linear'], rng=rng))
images = None
for j in range(1, 5):
    img = scipy.misc.imread('/home/tfinn/Projects/pyextremelm/examples/images/'
                            'Image_Wkm_Aktuell_{0:d}.jpg'.format(j),
                            flatten=False, mode="RGB")
    data = np.asarray(img, dtype='float64')
    data = data[480:1480, 480:1480, :]/256
    data = scipy.misc.imresize(data, (256, 256), 'nearest')
    data = data - data.mean(axis=(0,1))
    S = data.std(axis=(0,1))
    data = data / (S + 0.000002)
    data = data.reshape(1, 256, 256, 3)
    if images is None:
        images = data
    else:
        images = np.concatenate((images, data), axis=0)

img_ = images.transpose(0, 3, 1, 2)
images = images.reshape(-1, 3)
elm.fit(img_)
#print(elm.layers[-1].layers[0].weights['input'].transpose(0,2,3,1))
# print(elm.layers[-1].layers[1].weights['input'][0])
# print(np.min(elm.layers[-1].layers[0].weights['input']), np.max(elm.layers[-1].layers[0].weights['input']))
# print(np.min(elm.layers[-1].layers[1].weights['input']), np.max(elm.layers[-1].layers[1].weights['input']))


filtered_img = elm.predict(img_)
filtered_img_ = filtered_img.reshape((-1, output_layer))
filtered_img_ = filtered_img_ - filtered_img_.mean(axis=0)
filtered_img_ = filtered_img_ / (filtered_img_.std(axis=0) + 0.000001)
ax = plt.subplot(1,1,1)
plt.hist2d(filtered_img_[:, 0], filtered_img_[:, 1], bins=1000,
           normed=LogNorm)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
plt.colorbar()
plt.show()

# plot original image and first and second components of output
w = int(np.sqrt(output_layer + 1))
h = int(np.ceil((output_layer + 1) / w))
for n in range(4):
    plt.subplot(h, w, 1)
    plt.axis('off')
    plt.imshow(
        scipy.misc.imread(
            '/home/tfinn/Projects/pyextremelm/examples/images/'
            'Image_Wkm_Aktuell_{0:d}.jpg'.format(n + 1),
            flatten=False, mode="RGB")[480:1480, 480:1480, :] / 256,
        interpolation='none', cmap='gray')
    for i in range(output_layer):
        plt.subplot(h, w, i + 2)
        plt.axis('off')
        plt.imshow(filtered_img[n, i, :, :], interpolation='none',
                   cmap='gray')
    plt.show()
count = 0
for batch in image_iterator(50):
    elm.update(batch)
    count += 1
    print('Iteration: {0:d}'.format(count))
    print(elm.layers[-1].weights['input'].sum(axis=(1, 2, 3)))
    if count%10 == 0:
        filtered_img = elm.predict(img_)
        filtered_img_ = filtered_img.reshape((-1, output_layer))
        filtered_img_ = filtered_img_ - filtered_img_.mean(axis=0)
        filtered_img_ = filtered_img_ / (filtered_img_.std(axis=0) + 0.000001)
        ax = plt.subplot()
        plt.hist2d(filtered_img_[:, 0], filtered_img_[:, 1], bins=1000,
                   normed=LogNorm)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        plt.colorbar()
        plt.show()

        # plot original image and first and second components of output
        w = int(np.sqrt(output_layer + 1))
        h = int(np.ceil((output_layer + 1) / w))
        for n in range(4):
            plt.subplot(h, w, 1);
            plt.axis('off');
            plt.imshow(
                scipy.misc.imread(
                    '/home/tfinn/Projects/pyextremelm/examples/images/'
                            'Image_Wkm_Aktuell_{0:d}.jpg'.format(n + 1),
                    flatten=False, mode="RGB")[480:1480, 480:1480, :] / 256,
                interpolation='none', cmap='gray')
            for i in range(output_layer):
                plt.subplot(h, w, i + 2);
                plt.axis('off');
                plt.imshow(filtered_img[n, i, :, :], interpolation='none',
                           cmap='gray')
            plt.show()

# # Clustering test
# filtered_img = filtered_img.transpose(0,2,3,1)
# #filtered_img = np.concatenate((filtered_img, images), axis=3)
# filtered_img = filtered_img-np.mean(filtered_img, axis=(0,1,2))
# filtered_img = filtered_img/(np.std(filtered_img, axis=(0,1,2))+0.000001)
# b, w, h, c = filtered_img.shape
# filtered_img = filtered_img.reshape(-1, c)
#
# try:
#     ax = plt.subplot()
#     plt.hist2d(filtered_img[:,0], filtered_img[:,1], bins=1000)
#     ax.set_xlim(-2, 2)
#     ax.set_ylim(-2, 2)
#     plt.colorbar()
#     plt.show()
# except:
#     plt.hist(filtered_img[:,0], bins=100, normed=LogNorm)
#     plt.colorbar()
#     plt.show()
#
#
# kmeans = KMeans(2)
# kmeans.fit(filtered_img)
# labels = kmeans.labels
# labels = labels.labels.reshape((b, w, h))
# for n in range(4):
#     plt.subplot(1, 2, 1); plt.axis('off'); plt.imshow(
#         scipy.misc.imread('images/Image_Wkm_Aktuell_{0:d}.jpg'.format(n+1),
#                           flatten=False, mode="RGB")[480:1480, 480:1480, :]/256,
#         interpolation='none', cmap='gray')
#     plt.subplot(1, 2, 2); plt.axis('off'); plt.imshow(labels[n], interpolation='none')
#     plt.show()
#
