#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wen Jan 26 10:47 2022
@author: lou
"""

import numpy as np
import pandas as pd
import os
import h5py

def get_complementary_strand_OH(arr):
    '''
    Takes a DNA sequence in one hot format and return the reverse
    complement of the sequence.
    '''
    arr = arr[::-1]
    def complement(nuc):
        if np.array_equal(nuc, [0,0,0,1]):
            res = [0,0,1,0]
        elif np.array_equal(nuc, [0,0,1,0]):
            res = [0,0,0,1]
        elif np.array_equal(nuc, [0,1,0,0]):
            res = [1,0,0,0]
        elif np.array_equal(nuc, [1,0,0,0]):
            res = [0,1,0,0]
        else:
            res = nuc
        return res

    return np.apply_along_axis(complement, axis=1, arr=arr)

def get_complementary_strand_seq(arr):
    '''
    Takes a DNA sequence in one hot format and return the reverse
    complement of the sequence.
    '''
    arr_inv = arr[::-1]
    def complement(nuc):
        if nuc == 1:
            comp = 2
        elif nuc == 2:
            comp = 1
        elif nuc == 3:
            comp = 4
        elif nuc == 4:
            comp = 3
        else:
            comp = nuc
        return comp

    complement_vec = np.vectorize(complement)
    res = complement_vec(arr_inv)
    return res

def get_GC_content(arr):
    return round(((len(arr[arr==3])+len(arr[arr==4]))/len(arr))*100,2)

def one_hot_encoding_seq(X):
    '''
    Takes DNA a sequence in number format and return the corresponding
    one hot encoded sequence.
    '''
    bool = (np.arange(1,5) == X[...,None])
    X_one_hot = bool.astype('int8')
    return X_one_hot

def load_data(species, window, fast=False, padding=True):
    if fast:
        data = load_data_OH(species, window, padding)
    else:
        data = load_data_seq(species, window, padding)
    return data

def load_data_OH(species, window, padding=True):
    '''
    Takes a species ID and a window size and load the corresponding
    data to give to a generator. The data will be one hot encoded
    and loaded in a vectorized sliding window for better performances.
    '''
    data = {}
    for s in species:
        data[s] = {}
        files = os.listdir(f'Data/DNA/{s}/one_hot')
        print(s)
        for file in files:
            f = np.load(f'Data/DNA/{s}/one_hot/{file}')
            if padding:
                f = np.append(np.zeros((window//2,4),dtype='int8'), f, axis=0)
                f = np.append(f, np.zeros((window//2,4),dtype='int8'), axis=0)
            id = file.replace('.npy','')
            id = id.replace('chr','')
            sliding_window = sliding_window_view(f, (window,4), axis=(0,1))
            data[s][id] = sliding_window.reshape(sliding_window.shape[0],
                                                 sliding_window.shape[2], 
                                                 sliding_window.shape[3],1)
    return data

def load_data_seq(species, window, padding=True):
    '''
    Alternative data loader.
    Slower but memory efficient.
    '''
    data = {}
    for s in species:
        data[s] = {}
        files = os.listdir(f'Data/DNA/{s}/hdf5')
        print(s)
        for file in files:
            f = h5py.File(f'Data/DNA/{s}/hdf5/{file}','r')
            DNA = np.array(f['data'])
            f.close()
            DNA = DNA.reshape(DNA.shape[0],)
            DNA = DNA.astype('int8')
            if padding:
                DNA = np.append(np.zeros(window//2,dtype='int8'), DNA)
                DNA = np.append(DNA, np.zeros(window//2,dtype='int8'))
            id = file.replace('.hdf5','')
            id = id.replace('chr','')                                
            data[s][id] = sliding_window_view(DNA, window)
    return data

def load_data_one_chr(species, chr, window, reverse=False, padding=True):
    '''
    Takes a species ID, a chromosome ID and a window size and load the 
    corresponding data to give to a prediction generator. The data will 
    be one hot encoded and loaded in a vectorized sliding window for 
    better performances.
    '''
    f = np.load(f'Data/DNA/{species}/one_hot/chr{chr}.npy')
    if padding:
        f = np.append(np.zeros((window//2,4),dtype='int8'), f, axis=0)
        f = np.append(f, np.zeros((window//2,4),dtype='int8'), axis=0)
    if reverse:
        f = get_complementary_strand_OH(f)
    sliding_window = sliding_window_view(f, (window,4), axis=(0,1))
    data = sliding_window.reshape(sliding_window.shape[0],
                                  sliding_window.shape[2], 
                                  sliding_window.shape[3],1)
    return data

def load_data_full(window):
    gene_start = np.load('Predictions/multi3_gene_start_pred.npy')
    gene_stop = np.load('Predictions/multi3_gene_stop_pred.npy')
    exon_start = np.load('Predictions/multi3_exon_start_pred.npy')
    exon_stop = np.load('Predictions/multi3_exon_stop_pred.npy')
    data = np.array([gene_start, gene_stop, exon_start, exon_stop])
    data = np.reshape(data.flatten(order='F'), (data.shape[1],data.shape[0]))
    data =  np.append(np.zeros((window//2,4),dtype='float32'), data, axis=0)
    data = np.append(data, np.zeros((window//2,4),dtype='float32'), axis=0) 
    data = sliding_window_view(data, (window,4), axis=(0,1))
    data = data.reshape(data.shape[0],
                        data.shape[2], 
                        data.shape[3],1)
    return data
    
def load_data_multi_species(species_list, window, step):

    data = [np.zeros((window//2,4),dtype='int8')]
    indexes = np.array([], dtype=int)
    labels = np.array([], dtype='int8')
    total_len = 0

    for species in species_list:

        files = os.listdir(f'Data/DNA/{species}/one_hot')
        print(species)
        if len(species_list) == 1:
            files.sort()
            files.remove(files[0])

        for f in files:
            print(f)
            chr = np.load(f'Data/DNA/{species}/one_hot/{f}')
            data.append(chr)
            data.append(np.zeros((window//2,4),dtype='int8'))

            lab = np.load(f'Data/Positions/{species}/strand+/{f}')
            labels = np.append(labels, lab)
            labels = np.append(labels, np.zeros(window//2, dtype='int8'))

            indexes = np.append(indexes, np.arange(total_len, total_len+len(chr)+1, step= step))
            total_len += len(chr) + (window // 2)

    data = np.concatenate(data, axis=0)
    data = sliding_window_view(data, (window,4), axis=(0,1))
    data = data.reshape(data.shape[0],
                        data.shape[2], 
                        data.shape[3],1)

    labels = labels[:-(window//2)]
    tmp = labels[indexes]
    ratio = len(tmp[tmp==0])/len(tmp[tmp==1])

    return data, labels, indexes, ratio

def load_data_HS(window, step):

    files = os.listdir(f'Data/DNA/HS37/one_hot')
    files.sort()
    files.remove(files[0]) # <- PRED chr1

    data = [np.zeros((window//2,4),dtype='int8')]
    indexes = np.array([], dtype=int)
    labels = np.array([], dtype='int8')
    total_len = 0

    for f in files:
        print(f)
        chr = np.load(f'Data/DNA/HS37/one_hot/{f}')
        data.append(chr)
        data.append(np.zeros((window//2,4),dtype='int8'))

        lab = np.load(f'Data/Positions/HS37/strand+/{f}')
        labels = np.append(labels, lab)
        labels = np.append(labels, np.zeros(window//2, dtype='int8'))

        indexes = np.append(indexes, np.arange(total_len, total_len+len(chr)+1, step= step))
        total_len += len(chr) + (window // 2)

    data = np.concatenate(data, axis=0)
    data = sliding_window_view(data, (window,4), axis=(0,1))
    data = data.reshape(data.shape[0],
                        data.shape[2], 
                        data.shape[3],1)

    labels = labels[:-(window//2)]
    tmp = labels[indexes]
    ratio = len(tmp[tmp==0])/len(tmp[tmp==1])

    return data, labels, indexes, ratio


def sliding_window_view(x, window_shape, axis=None, *,
                        subok=False, writeable=False):
    '''
    Takes an numpy array and a window size and return a vectorized
    sliding window. 
    '''
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        axis = np.core.numeric.normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return np.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape,
                      subok=subok, writeable=writeable)

