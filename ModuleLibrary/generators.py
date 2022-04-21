#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wen Feb 23 9:53 2022
@author: lou
"""

import tensorflow as tf
import numpy as np
from ModuleLibrary.utils import traduction, one_hot_prot

class Generator_Features(tf.keras.utils.Sequence):

    def __init__(self, indexes, data, labels, batch_size,
                 window, shuffle=True, data_augment=False):
        self.dim = (window,4)
        self.list_IDs = indexes
        self.batch_size = batch_size
        self.labels = labels
        self.data = data
        self.shuffle = shuffle
        self.data_augment = data_augment
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
        Y = np.empty((self.batch_size,2), dtype='int8')
        for i, ID in enumerate(indexes):
            X[i,] = self.data[ID]
            Y[i] = self.labels[ID]
        return X, Y

    def data_shift(self): 
        self.list_IDs = self.list_IDs + 2

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.data_augment:
            self.data_shift()
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

class Generator_Protein(tf.keras.utils.Sequence):

    def __init__(self, indexes, data, labels, 
                 batch_size, window, mode, shuffle=True):
        self.dim = (window,21)
        self.list_IDs = indexes
        self.batch_size = batch_size
        self.labels = labels
        self.data = data
        self.mode = mode
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
        Y = np.empty((self.batch_size,), dtype='int8')
        for i, ID in enumerate(indexes):
            lab = self.labels[ID]
            data = self.data[ID]
            if self.mode == 'randperm' and lab == 0:
                data = np.random.permutation(data)
            Y[i] = lab
            X[i,] = data
            
        return X, Y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
class Generator_Coverage(tf.keras.utils.Sequence):

    def __init__(self, indexes, data, labels, batch_size, mode,
                 window, shuffle=True, data_augment=False):
        self.dna_dim = (window, 4, 1)
        self.pred_dim = (window, mode*2, 1)
        self.mode = mode
        self.list_IDs = indexes
        self.batch_size = batch_size
        self.labels = labels
        self.data = data
        self.shuffle = shuffle
        self.data_augment = data_augment
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        DNA, PRED, Y = self.__data_generation(list_IDs_temp)
        if self.mode > 0:
            output = [DNA, PRED]
        else:
            output = DNA 
        return output, Y

    def __data_generation(self, indexes):
        DNA = np.empty((self.batch_size, *self.dna_dim), dtype='int8')
        PRED = np.empty((self.batch_size, *self.pred_dim), dtype=float)
        Y = np.empty((self.batch_size), dtype='int8')

        for i, ID in enumerate(indexes):
            DNA[i,] = self.data[0][ID]
            Y[i] = self.labels[ID]
            if self.mode > 0:
                PRED[i,] = self.data[1][ID]

        return DNA, PRED, Y

    def data_shift(self): # Data augment
        self.list_IDs = self.list_IDs + 2

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.data_augment:
            self.data_shift()
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

class Generator_Coverage_from_dna(tf.keras.utils.Sequence):

    def __init__(self, indexes, data, labels, batch_size,
                 window, shuffle=True, data_augment=False):
        self.dim = (window,4)
        self.list_IDs = indexes
        self.batch_size = batch_size
        self.labels = labels
        self.data = data
        self.shuffle = shuffle
        self.data_augment = data_augment
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        DNA, Y = self.__data_generation(list_IDs_temp)
        return DNA, Y

    def __data_generation(self, indexes):
        DNA = np.empty((self.batch_size, *self.dim), dtype='int8')
        Y = np.empty((self.batch_size), dtype='int8')
        for i, ID in enumerate(indexes):
            DNA[i,] = self.data[ID]
            Y[i] = self.labels[ID]
        return DNA, Y

    def data_shift(self): # Data augment
        self.list_IDs = self.list_IDs + 2

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.data_augment:
            self.data_shift()
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

class Generator_Prediction(tf.keras.utils.Sequence):

    def __init__(self, data, batch_size, window):
        self.dim = (window,4)
        self.batch_size = batch_size
        self.data = data
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size)) + 1

    def get_last_batch_size(self):
        last_batch_size = len(self.data) - (self.batch_size * (self.__len__() - 1))
        return last_batch_size

    def __getitem__(self, index):
        if index == self.__len__() - 1:
            batch_indexes = self.indexes[index*self.batch_size:(index*self.batch_size)+self.get_last_batch_size()]
        else:
            batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = self.__data_generation(batch_indexes)
        return X

    def __data_generation(self, batch_indexes):
        X = np.empty((len(batch_indexes), *self.dim, 1), dtype='int8')
        for i, ID in enumerate(batch_indexes):
            X[i,] = self.data[ID]
        return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))

class Generator_Prediction_prot(tf.keras.utils.Sequence):

    def __init__(self, data, batch_size, window):
        self.dim = (window,21,1)
        self.batch_size = batch_size
        self.data = data
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size)) + 1

    def get_last_batch_size(self):
        last_batch_size = len(self.data) - (self.batch_size * (self.__len__() - 1))
        return last_batch_size

    def __getitem__(self, index):
        if index == self.__len__() - 1:
            batch_indexes = self.indexes[index*self.batch_size:(index*self.batch_size)+self.get_last_batch_size()]
        else:
            batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X = self.__data_generation(batch_indexes)
        return X

    def __data_generation(self, batch_indexes):
        X = np.empty((len(batch_indexes), *self.dim), dtype='int8')
        for i, ID in enumerate(batch_indexes):
            x = self.data[ID]
            x = traduction(x)
            x = one_hot_prot(x)
            x = x.reshape(x.shape[0], x.shape[1],1)
            X[i,] = x
        return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data))

class Generator_Prediction_Coverage(tf.keras.utils.Sequence):
   
    def __init__(self, data, batch_size, window, mode):
        self.dna_dim = (window, 4, 1)
        self.pred_dim = (window, mode*2, 1)
        self.batch_size = batch_size
        self.data = data
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.data[0]) / self.batch_size)) + 1

    def get_last_batch_size(self):
        last_batch_size = len(self.data[0]) - (self.batch_size * (self.__len__() - 1))
        return last_batch_size

    def __getitem__(self, index):
        if index == self.__len__() - 1:
            batch_indexes = self.indexes[index*self.batch_size:(index*self.batch_size)+self.get_last_batch_size()]
        else:
            batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        DNA, PRED = self.__data_generation(batch_indexes)
        return [DNA, PRED]

    def __data_generation(self, batch_indexes):
        DNA = np.empty((self.batch_size, *self.dna_dim), dtype='int8')
        PRED = np.empty((self.batch_size, *self.pred_dim), dtype=float)
        for i, ID in enumerate(batch_indexes):
            DNA[i,] = self.data[0][ID]
            PRED[i,] = self.data[1][ID]
        return DNA, PRED

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.data[0]))