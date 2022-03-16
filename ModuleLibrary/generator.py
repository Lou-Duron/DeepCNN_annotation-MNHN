#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wen Feb 23 9:53 2022
@author: lou
"""

import tensorflow as tf
import numpy as np
from ModuleLibrary.utils import get_complementary_strand_OH, one_hot_encoding_seq, get_complementary_strand_seq

class DataGenerator(tf.keras.utils.Sequence):
    '''
    Generator for model training
    '''
    def __init__(self, list_IDs, labels, data, batch_size, window,
                  shuffle=True, fast=False):
        self.dim = (window,4)
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.data = data
        self.shuffle = shuffle
        self.fast = fast
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, 1), dtype='int8')
        Y = np.empty((self.batch_size), dtype='int8')
        for i, ID in enumerate(list_IDs_temp):
            infos = ID.split('_')
            species = infos[0]
            chromosome = infos[1]
            position = int(infos[2])
            strand = int(infos[3])
            seq = self.data[species][chromosome][position - 1]
            if self.fast:
                if strand == 3:
                    seq = get_complementary_strand_OH(seq)
            else:
                if strand == 3:
                    seq = get_complementary_strand_seq(seq)
                seq = one_hot_encoding_seq(seq)
                seq = seq.reshape(seq.shape[0], seq.shape[1], 1)
            X[i,] = seq
            Y[i] = self.labels[ID]
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

class DataGeneratorFull(tf.keras.utils.Sequence):

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