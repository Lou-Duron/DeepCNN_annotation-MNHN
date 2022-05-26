#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Wen Jan 26 2022
@author: Lou Duron

This module contains all sort of utils function used
in other modules.
"""

import numpy as np

def get_reverse_complement_from_seq(arr):
    '''
    Takes a DNA sequence in 1D format and return the reverse
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

def get_reverse_complement_from_OH(arr):
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

def OH_to_DNA(arr):
    '''
    Takes a DNA sequence in one hot format and return the same
    sequence in fasta format.
    '''
    res = np.array([])
    for nuc in arr:
        if np.array_equal(nuc, [0,0,0,1]):
            res = np.append(res, 'c')
        elif np.array_equal(nuc, [0,0,1,0]):
            res = np.append(res, 'g')
        elif np.array_equal(nuc, [0,1,0,0]):
            res = np.append(res, 't')
        elif np.array_equal(nuc, [1,0,0,0]):
            res = np.append(res, 'a')
        else:
            res = np.append(res, '-')
    return res

def get_GC_content(arr):
    '''
    Takes a DNA sequence in 1D format and return the GC content.
    '''
    return round(((len(arr[arr==3])+len(arr[arr==4]))/len(arr))*100,2)

def one_hot_DNA(arr):
    '''
    Takes DNA a sequence in 1D format and return the corresponding
    one hot encoded sequence.
    '''
    bool = (np.arange(1,5) == arr[...,None])
    arr_OH = bool.astype('int8')
    return arr_OH

def one_hot_prot(arr):
    '''
    Takes protein a sequence in 1D format and return the corresponding
    one hot encoded sequence.
    '''
    bool = (np.arange(0,21) == arr[...,None])
    arr_one_hot = bool.astype('int8')
    return arr_one_hot

def sliding_window_view(x, window_shape, axis=None, *, subok=False, writeable=False):
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

def traduction(arr):
    '''
    Takes DNA a sequence in 1D format and return the corresponding
    protein sequence in 1D format.
    STOP:0 K:1 N:2 I:3 R:4 S:5 T:6 Y:7 L:8 F:9 C:10
    W:11 E:12 D:13 V:14 G:15 A:16 Q:17 H:18 P:19 None:20
    '''
    dic = { 1:{1:{1:1, 2:2, 3:1, 4:2},
              2:{1:3, 2:3, 3:20, 4:3},
              3:{1:4, 2:5, 3:4, 4:5},
              4:{1:6, 2:6, 3:6, 4:6}},
            2:{1:{1:0, 2:7, 3:0, 4:7},
              2:{1:8, 2:9, 3:8, 4:9},
              3:{1:0, 2:10, 3:11, 4:10},
              4:{1:5, 2:5, 3:5, 4:5}},
            3:{1:{1:12, 2:13, 3:12, 4:13},
              2:{1:14, 2:14, 3:14, 4:14},
              3:{1:15, 2:15, 3:15, 4:15},
              4:{1:16, 2:16, 3:16, 4:16}},
            4:{1:{1:17, 2:18, 3:17, 4:18},
              2:{1:8, 2:8, 3:8, 4:8},
              3:{1:4, 2:4, 3:4, 4:4},
              4:{1:19, 2:19, 3:19, 4:19}}}
    arr = arr[:len(arr) - len(arr) % 3]
    arr = arr.reshape((-1,3))
    res = []
    for codon in arr:
        if np.any(codon == 0):
            res.append(20)
        else:
            res.append(dic[codon[0]][codon[1]][codon[2]])
    return np.array(res)

def traduction_3_frames(seq):
    '''
    Takes DNA a sequence in 1D format and return a list of the corresponding
    protein sequences in 1D format in all framme possible.
    '''
    frames, res = [], []

    f1 = seq[:len(seq) - len(seq) % 3]
    frames.append(f1.reshape((-1,3)))

    f2 = seq[1:]
    f2 = f2[:(len(f2) - len(f2) % 3)]
    frames.append(f2.reshape((-1,3)))

    f3 = seq[2:]
    f3 = f3[:(len(f3) - len(f3) % 3)]
    frames.append(f3.reshape((-1,3)))

    for frame in frames:
        res.append(traduction(frame))
    return res

def encode_aa(seq):
    '''
    Takes protein a sequence in fasta format and return a
    encoded array in 1D format.
    '''
    dic = {'K':1, 'N':2, 'I':3, 'R':4, 'S':5, 'T':6, 'Y':7, 'L':8, 'F':9, 'C':10,'W':11,
           'E':12, 'D':13, 'V':14, 'G':15, 'A':16, 'Q':17, 'H':18, 'P':19, 'M':20}
    def encode(aa):
        return dic[aa]
    encode_vec = np.frompyfunc(encode, 1,1)
    return encode_vec(seq).astype('int8')



 

