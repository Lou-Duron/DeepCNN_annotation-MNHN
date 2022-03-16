#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wen Feb 23 9:53 2022
@author: lou
"""

import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, indexes, data, labels, 
                 batch_size, window, shuffle=True):
        self.dim = (window,4)
        self.list_IDs = indexes
        self.batch_size = batch_size
        self.labels = labels
        self.data = data
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y

    def __data_generation(self, indexes):
        X = np.empty((self.batch_size, *self.dim, 1), dtype='int8')
        Y = np.empty((self.batch_size), dtype='int8')
        for i, ID in enumerate(indexes):
            X[i,] = self.data[ID]
            Y[i] = self.labels[ID]
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

class PredGenerator(tf.keras.utils.Sequence):
    '''
    Generator for prediction
    '''
    def __init__(self, data, batch_size, window):
        self.dim = (window,4)
        self.batch_size = batch_size
        self.data = data
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = self.__data_generation(batch_indexes)
        return X

    def __data_generation(self, batch_indexes):
        X = np.empty((self.batch_size, *self.dim, 1), dtype='int8')
        for i, ID in enumerate(batch_indexes):
            X[i,] = self.data[ID]
        return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))