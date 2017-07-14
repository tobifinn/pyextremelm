# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:14:25 2016

@author: tfinn
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import scipy.ndimage

from pyextremelm.builder import ExtremeLearningMachine
from pyextremelm.builder.layers.convolution import ELMLRF
from pyextremelm.builder.layers.pooling import ELMPool
from pyextremelm.builder.layers.util import ELMNormalize

rng = np.random.RandomState(42)
output_layer = 20

elm = ExtremeLearningMachine()
elm.add_layer(ELMLRF(40, (3,3), pad=(1,1), activation='relu', ortho=True, rng=rng))
elm.add_layer(ELMNormalize())
elm.add_layer(ELMLRF(80, (3,3), pad=(1,1), activation='relu', ortho=True, rng=rng))
elm.add_layer(ELMNormalize())
elm.add_layer(ELMLRF(160, (3,3), pad=(1,1), activation='relu', ortho=True, rng=rng))
elm.add_layer(ELMPool('squareroot'))
elm.add_layer(ELMNormalize())
elm.add_layer(ELMLRF(320, (3,3), pad=(1,1), activation='relu', ortho=True, rng=rng))
elm.add_layer(ELMNormalize())
elm.add_layer(ELMLRF(160, (3,3), pad=(1,1), activation='relu', ortho=True, rng=rng))
elm.add_layer(ELMNormalize())
elm.add_layer(ELMLRF(80, (3,3), pad=(1,1), activation='relu', ortho=True, rng=rng))
elm.add_layer(ELMNormalize())
elm.add_layer(ELMLRF(20, (3,3), pad=(1,1), activation='sigmoid', ortho=True, rng=rng))
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


filtered_img_ = filtered_img.reshape((-1, output_layer))
filtered_img_ = filtered_img_ - filtered_img_.mean(axis=0)
filtered_img_ = filtered_img_/(filtered_img_.std(axis=0)+0.000001)
try:
    w = int(np.sqrt(output_layer))
    h = int(np.ceil((output_layer)/w))
    for i in range(output_layer):
        plt.subplot(h, w, i + 1)
        beta = elm.layers[-1].weights['input'][i,:,:,:]
        plt.hist(beta.reshape((-1)))
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(e)
try:
    w = int(np.sqrt(output_layer))
    h = int(np.ceil((output_layer)/w))
    for i in range(output_layer):
        plt.subplot(h, w, i + 1)
        plt.axis('off')
        plt.imshow(np.mean(elm.layers[-1].weights['input'][i,:,:,:], axis=0), interpolation='none', cmap='gray')
    plt.show()
except Exception as e:
    print(e)


# plot original image and first and second components of output
w = int(np.sqrt(output_layer+1))
h = int(np.ceil((output_layer+1)/w))
print((w, h))

for n in range(4):
    fig, ax = plt.subplots(nrows=h, ncols=w)

    plt.axis('off')
    ax[0, 0].imshow(
        scipy.misc.imread('images/Image_Wkm_Aktuell_{0:d}.jpg'.format(n+1),
                          flatten=False, mode="RGB")[480:1480, 480:1480, :]/256,
         interpolation='none', cmap='gray')
    for i in range(output_layer):
        ax[int((i+1)/w), (i+1)%w].imshow(filtered_img[n, i, :, :], interpolation='none', cmap='gray')
    #plt.show()
    fig.tight_layout()
    fig.savefig('lrf_{0:d}.pdf'.format(n))

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
