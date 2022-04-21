#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wen Jan 26 10:47 2022
@author: lou
"""

import h5py
import numpy as np
import os
import pandas as pd

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

def OH_to_DNA(arr):
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

def one_hot_prot(seq):
    bool = (np.arange(0,21) == seq[...,None])
    seq_one_hot = bool.astype('int8')
    return seq_one_hot

def load_data_one_chr(species, chr, window, reverse=False):
    '''
    Takes a species ID, a chromosome ID and a window size and load the 
    corresponding data to give to a prediction generator. The data will 
    be one hot encoded and loaded in a vectorized sliding window for 
    better performances.
    '''
    f = np.load(f'Data/DNA/{species}/one_hot/chr{chr}.npy')
    f = np.append(np.zeros((window//2,4),dtype='int8'), f, axis=0)
    f = np.append(f, np.zeros((window//2,4),dtype='int8'), axis=0)
    if reverse:
        f = get_complementary_strand_OH(f)
    sliding_window = sliding_window_view(f, (window,4), axis=(0,1))
    data = sliding_window.reshape(sliding_window.shape[0],
                                  sliding_window.shape[2], 
                                  sliding_window.shape[3],1)
    return data

def load_data_one_chr_prot(species, chr, window, reverse=False):

    h5 = h5py.File(f'Data/DNA/{species}/hdf5/chr{chr}.hdf5','r')
    dna = np.array(h5['data']).astype('int8')
    dna = dna.reshape(dna.shape[0])
    dna = np.append(np.zeros(((window*3)//2),dtype='int8'), dna, axis=0)
    dna = np.append(dna, np.zeros(((window*3)//2),dtype='int8'), axis=0)
    data = sliding_window_view(dna, window*3)
    return data

def load_data_one_chr_coverage(species, chr, window, mode=1):
    '''
    Takes a species ID, a chromosome ID and a window size and load the 
    corresponding data to give to a prediction generator. The data will 
    be one hot encoded and loaded in a vectorized sliding window for 
    better performances.
    '''
    data = []

    f = np.load(f'Data/DNA/{species}/one_hot/chr{chr}.npy').astype('int8')
    f = np.append(np.zeros((window//2,4),dtype='int8'), f, axis=0)
    f = np.append(f, np.zeros((window//2,4),dtype='int8'), axis=0)

    sliding_window = sliding_window_view(f, (window,4), axis=(0,1))
    dna = sliding_window.reshape(sliding_window.shape[0],
                                  sliding_window.shape[2], 
                                  sliding_window.shape[3],1)
    data.append(dna)

    pred_list = []
    pred_list.append(np.load(f'Predictions/{species}_gene_start/chr{chr}.npy'))
    pred_list.append(np.load(f'Predictions/{species}_gene_stop/chr{chr}.npy'))
    if mode > 1:
        pred_list.append(np.load(f'Predictions/{species}_exon_start/chr{chr}.npy'))
        pred_list.append(np.load(f'Predictions/{species}_exon_stop/chr{chr}.npy'))
    if mode > 2:
        pred_list.append(np.load(f'Predictions/{species}_rna_start/chr{chr}.npy'))
        pred_list.append(np.load(f'Predictions/{species}_rna_stop/chr{chr}.npy'))

    pred = np.array(pred_list)
    pred = np.reshape(pred.flatten(order='F'), (pred.shape[1],pred.shape[0]))
    pred = np.append(np.zeros((window//2,mode*2),dtype='float32'), pred, axis=0)
    pred = np.append(pred, np.zeros((window//2,mode*2),dtype='float32'), axis=0)

    pred = sliding_window_view(pred, (window,mode*2), axis=(0,1))
    pred = pred.reshape(pred.shape[0],
                      pred.shape[2], 
                      pred.shape[3],1)  
    data.append(pred)

    return data

def load_data_features(species_list, window, ratio, validation, features, mode, chr_nb):

    data = np.zeros((window//2,4),dtype='int8')
    indexes = np.array([], dtype=int)
    labels = np.array([], dtype='int8')
    total_len = 0

    for species in species_list:
        files = os.listdir(f'Data/DNA/{species}/one_hot')
        annot = pd.read_csv(f'Data/Annotations/{species}/{features}.csv', sep = ',')
        annot = annot.drop_duplicates(subset=['chr', 'stop', 'start', 'strand'], keep='last') 
        if len(species_list) == 1:
            files.sort()
            files.remove(files[0])
        print('')

        for i, f in enumerate(files):
            print(f'{species} : {i+1}/{len(files)}', end = '\r')
            
            chr = np.load(f'Data/DNA/{species}/one_hot/{f}')
            data = np.append(data, chr, axis=0)
            data = np.append(data, np.zeros((window//2,4),dtype='int8'), axis=0)

            ANNOT = annot[(annot.chr == f.replace('.npy','') )]
            ANNOT = ANNOT[(ANNOT.strand == '+')]
        
            if mode == 'start':
                feature_pos = np.unique(ANNOT['start'].values)
            elif mode == 'stop':
                feature_pos = np.unique(ANNOT['stop'].values)
            feature_pos = feature_pos - 1

            lab = np.zeros(len(chr), dtype='int8')

            def fill(x):
                lab[x] = 1
            fill_vec = np.frompyfunc(fill, 1,0)
            fill_vec(feature_pos)

            labels = np.append(labels, lab)
            labels = np.append(labels, np.zeros(window//2, dtype='int8'))

            indexes_chr = np.arange(len(chr))
            indexes_to_remove = np.array([], dtype = int)
            positions = np.unique(feature_pos)

            for i in positions:
                range_to_remove = np.arange(i - window, i + window + 1, dtype = int)
                indexes_to_remove = np.append(indexes_to_remove, range_to_remove)
            indexes_to_remove = indexes_to_remove[indexes_to_remove < len(chr)-1]
            indexes_chr = np.delete(indexes_chr, np.unique(indexes_to_remove))

            np.random.shuffle(indexes_chr)

            indexes_chr = np.append(positions, indexes_chr[:len(positions)*ratio])

            

            indexes_chr = indexes_chr + total_len
            indexes = np.append(indexes, indexes_chr)
            total_len += len(chr) + (window // 2)

    data = sliding_window_view(data, (window,4), axis=(0,1))
    data = data.reshape(data.shape[0],
                        data.shape[2], 
                        data.shape[3],1)
 
    labels = labels[:-(window//2)]

    np.random.shuffle(indexes)

    train_indexes = indexes[int(len(indexes)*validation):]
    val_indexes = indexes[:int(len(indexes)*validation)]

    return data, labels, train_indexes, val_indexes
    
def load_data_gene_coverage(species_list, window, step, validation, mode, chr_nb, features):

    dna = np.zeros((window//2,4),dtype='int8')
    pred = np.zeros((window//2,mode*2),dtype='float32')
    indexes = np.array([], dtype=int)
    labels = np.array([], dtype='int8')
    total_len = 0

    for species in species_list:
        files = os.listdir(f'Data/DNA/{species}/one_hot')
        annot = pd.read_csv(f'Data/Annotations/{species}/{features}.csv', sep = ',')
        annot = annot.drop_duplicates(subset=['chr', 'stop', 'start', 'strand'], keep='last')
        if len(species_list) == 1:
            files.sort()
            files.remove(files[0])
        print('')

        for i, f in enumerate(files):
            print(f'{species} : {i+1}/{len(files)}', end = '\r')

            chr = np.load(f'Data/DNA/{species}/one_hot/{f}')
            dna = np.append(dna, chr, axis=0)
            dna = np.append(dna, np.zeros((window//2,4),dtype='int8'), axis=0)

            ANNOT = annot[(annot.chr == f.replace('.npy','') )]
            ANNOT = ANNOT[(ANNOT.strand == '+')]
            feature_start = ANNOT['start'].values 
            feature_start = feature_start - 1
            feature_stop = ANNOT['stop'].values 
            feature_stop = feature_stop - 1
            lab = np.zeros(len(chr), dtype='int8')
            def fill(x,y):
                for i in range(x,y+1):
                    lab[i] = 1
            fill_vec = np.frompyfunc(fill, 2, 0)
            fill_vec(feature_start, feature_stop)

            labels = np.append(labels, lab)
            labels = np.append(labels, np.zeros(window//2, dtype='int8'))

            indexes = np.append(indexes, np.arange(total_len, total_len+len(chr)+1, step= step))
            total_len += len(chr) + (window // 2)

            
            if mode > 0:
                pred_list = []
                pred_list.append(np.load(f'Predictions/{species}_gene_start/{f}'))
                pred_list.append(np.load(f'Predictions/{species}_gene_stop/{f}'))
                if mode > 1:
                    pred_list.append(np.load(f'Predictions/{species}_exon_start/{f}'))
                    pred_list.append(np.load(f'Predictions/{species}_exon_stop/{f}'))
                if mode > 2:
                    pred_list.append(np.load(f'Predictions/{species}_rna_start/{f}'))
                    pred_list.append(np.load(f'Predictions/{species}_rna_stop/{f}'))

                chr_pred = np.array(pred_list)
                chr_pred = np.reshape(chr_pred.flatten(order='F'), (chr_pred.shape[1],chr_pred.shape[0]))
                pred = np.append(pred, chr_pred, axis=0)
                pred = np.append(pred, np.zeros((window//2,mode*2),dtype='float32'), axis=0)
                
            if chr_nb > 0 and i == chr_nb - 1:
                break

    dna = sliding_window_view(dna, (window,4), axis=(0,1))
    dna = dna.reshape(dna.shape[0],
                      dna.shape[2], 
                      dna.shape[3],1)
    if mode > 0:
        pred = sliding_window_view(pred, (window,mode*2), axis=(0,1))
        pred = pred.reshape(pred.shape[0],
                        pred.shape[2], 
                        pred.shape[3],1)         

    labels = labels[:-(window//2)]
    tmp = labels[indexes]
    ratio = len(tmp[tmp==0])/len(tmp[tmp==1])

    

    np.random.shuffle(indexes)

    train_indexes = indexes[int(len(indexes)*validation):]
    val_indexes = indexes[:int(len(indexes)*validation)]

    return dna, pred, labels, ratio, train_indexes, val_indexes

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

def traduction(seq):

    seq = seq[:len(seq) - len(seq) % 3]

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

    #STOP:0 K:1 N:2 I:3 R:4 S:5 T:6 Y:7 L:8 F:9 C:10
    #W:11 E:12 D:13 V:14 G:15 A:16 Q:17 H:18 P:19 None:20

    seq = seq.reshape((-1,3))
    res = []
    for codon in seq:
        if np.any(codon == 0):
            res.append(20)
        else:
            res.append(dic[codon[0]][codon[1]][codon[2]])
    return np.array(res)

def traduction_3_frames(seq):

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

    #STOP:0 K:1 N:2 I:3 R:4 S:5 T:6 Y:7 L:8 F:9 C:10
    #W:11 E:12 D:13 V:14 G:15 A:16 Q:17 H:18 P:19  

    frames = []

    f1 = seq[:len(seq) - len(seq) % 3]
    frames.append(f1.reshape((-1,3)))

    f2 = seq[1:]
    f2 = f2[:(len(f2) - len(f2) % 3)]
    frames.append(f2.reshape((-1,3)))

    f3 = seq[2:]
    f3 = f3[:(len(f3) - len(f3) % 3)]
    frames.append(f3.reshape((-1,3)))

    res = []

    for frame in frames:
        frame_res = []
        for codon in frame:
            frame_res.append(dic[codon[0]][codon[1]][codon[2]])
        res.append(np.array(frame_res))
    return res

def encode_aa(seq):
    dic = {'K':1, 'N':2, 'I':3, 'R':4, 'S':5, 'T':6, 'Y':7, 'L':8, 'F':9, 'C':10,'W':11,
           'E':12, 'D':13, 'V':14, 'G':15, 'A':16, 'Q':17, 'H':18, 'P':19, 'M':20}
    def encode(aa):
        return dic[aa]
    encode_vec = np.frompyfunc(encode, 1,1)
    return encode_vec(seq).astype('int8')

def load_data_prot(win, validation, mode):

    data_pos = np.load(f'Uniprot/data_sprot_oh_w{win}.npy') # shape = (None, win , 21)
    
    if mode == 'randperm':    # random permut before split ?
        data = np.append(data_pos, data_pos, axis=0)
        data = data.reshape(data.shape[0],data.shape[1],data.shape[2],1)

    elif mode == 'fromHS37':
        files = os.listdir(f'Data/DNA/HS37/prot_OH')
        dna = np.empty((0,21), dtype='int8')
        indexes = np.array([], dtype=int)
        sum_lenght = 0
        for i, f in enumerate(files):
            print(f'{i+1}/{len(files)}', end='\r')
            chr = np.load(f'Data/DNA/HS37/prot_OH/{f}') # shape=(None, 21)
            dna = np.append(dna, chr, axis=0)
            indexes = np.append(indexes, np.arange(sum_lenght, sum_lenght + (len(chr) - win*3)))
            sum_lenght += len(chr)
        dna = sliding_window_view(dna, (win,21), axis=(0,1))  # shape = (None, win , 21)
        dna = dna.reshape(dna.shape[0], dna.shape[2], dna.shape[3])
        random_indexes = np.random.choice(indexes, len(data_pos), replace=False)
        data_neg = dna[random_indexes]
        data = np.append(data_pos, data_neg, axis=0)
        data = data.reshape(data.shape[0],data.shape[1],data.shape[2],1)
        
        
    elif mode == 'fullrand':
        pass
    else:
        print('Wrong mode')
        exit()
    labels = np.append(np.ones(len(data)//2, dtype='int8'), np.zeros(len(data)//2, dtype='int8'))
    indexes = np.arange(len(data),dtype='int32')
    np.random.shuffle(indexes)
    train_indexes = indexes[int(len(indexes)*validation):]
    val_indexes = indexes[:int(len(indexes)*validation)]

    return data, labels, train_indexes, val_indexes
    

 

