from __future__ import print_function
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:36:09 2018

@author: Ferdinand
"""

"""Functions for downloading and reading MNIST data."""

import gzip
import os
import urllib
import pandas as pd

import numpy as np
import cv2

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):

    def __init__(self, images, labels = None):
        
        self._num_examples = images.shape[0]
        
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images
    
    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            if self._labels is not None : self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if self._labels is not None :
            return self._images[start:end], self._labels[start:end]
        else:
            return self._images[start:end], None


class SemiDataSet(object):
    def __init__(self, labeled_images, labels, unlabeled_images):
        self.n_labeled = labeled_images.shape[0]
        self.n_unlabeled = unlabeled_images.shape[0]

        # Unlabled DataSet
        self.unlabeled_ds = DataSet(unlabeled_images)

        # Labeled DataSet
        self.labeled_ds = DataSet(labeled_images, labels)

    def next_batch(self, batch_size):
        unlabeled_images, _ = self.unlabeled_ds.next_batch(batch_size)
        if batch_size > self.n_labeled:
            labeled_images, labels = self.labeled_ds.next_batch(self.n_labeled)
        else:
            labeled_images, labels = self.labeled_ds.next_batch(batch_size)
#        images = np.vstack([labeled_images, unlabeled_images])
#        return images, labels
        return labeled_images, labels, unlabeled_images


def read_json(filename, with_labels = False):
    df = pd.read_json(filename)
    df['inc_angle'] = df['inc_angle'].replace('na', -1).astype(float)
    
    # Angles
    angles = df['inc_angle'].values
    
    # Bandes
    band1 = np.array([np.array(band).astype(np.float32).reshape([75,75]) for band in df["band_1"]])
    band2 = np.array([np.array(band).astype(np.float32).reshape([75,75]) for band in df["band_2"]])
    df = df.drop(['band_1', 'band_2'], axis=1)
#
    X = np.stack((band1, band2), axis=-1)
    
    if with_labels == True:
        Y = df['is_iceberg'].values
        Y = Y[:,np.newaxis]
        return X, angles, Y
    else:
        return X, angles
    
    
def import_data(file_labeled, file_unlabeled, image_shape):
    image_size = tuple(image_shape[:-1])
    x_l, ang_l, y_l = read_json(file_labeled, True)
    x_u, ang_u = read_json(file_unlabeled,  False)
    
    
    
    # resize
    resized = []
    for x in x_l:
        resized.append(cv2.resize(x,image_size))
    x_l = np.array(resized)
    
    resized = []
    for x in x_u:
        resized.append(cv2.resize(x,image_size))
    x_u = np.array(resized)
    
    # reshape
    x_l = np.reshape(x_l,[x_l.shape[0],np.prod(image_shape)])
    x_u = np.reshape(x_u,[x_u.shape[0],np.prod(image_shape)])
    
    # normalize
    x_l = x_l - np.min(x_l, axis = 1, keepdims = True)
    x_l = x_l/np.max(x_l, axis = 1, keepdims = True)
    
    x_u = x_u - np.min(x_u, axis = 1, keepdims = True)
    x_u = x_u/np.max(x_u, axis = 1, keepdims = True)
    
    # exclude computer generated images
    x_test = x_u
    ind = np.where(np.abs(np.round(ang_u,4) - ang_u) < 1e-10)[0]
    x_u = x_u[ind,:]
    
    # Add labeled images to unlabeled set
    x_u = np.concatenate((x_u,x_l), axis = 0)
    
    return x_l, y_l, x_u, x_test


def read_data_sets(file_labeled, file_unlabeled, image_shape = [75,75,2], val_ratio = 0.2):
    """ val_ratio : number between 0 and 1. Amount of data to be included in the validation set
        file_labeled : json file containing the labeled images
        file_unlabeled : json file containing the unlabeled images
        image_shape : shape of the image in the format [hight, width, n_chanels]"""
    
    class DataSets(object):
        pass
    data_sets = DataSets()
    
    # Extract data
    images_l, labels, images_u, test_set = import_data(file_labeled, file_unlabeled, image_shape)
    
    labels = dense_to_one_hot(labels, num_classes=2)
    
    # Split train and validation sets
    nb_val = int(np.floor(val_ratio*labels.shape[0]))
    ind = np.random.choice(np.arange(labels.shape[0]), nb_val)
    
    images_val = images_l[ind]
    labels_val = labels[ind]
    images_l = np.delete(images_l, ind, axis = 0)
    labels = np.delete(labels, ind, axis = 0)
    
    data_sets.train = SemiDataSet(images_l, labels, images_u)
    data_sets.validation = DataSet(images_val, labels_val)
    data_sets.test = DataSet(test_set)
    
    return data_sets
