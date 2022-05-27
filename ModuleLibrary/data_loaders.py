#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Wen Apr 20 2022
@author: Lou Duron

This module contains data loaders for training
"""

import numpy as np
import os
import pandas as pd
from ModuleLibrary.utils import get_reverse_complement_from_OH, sliding_window_view

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
        f = get_reverse_complement_from_OH(f)
    sliding_window = sliding_window_view(f, (window,4), axis=(0,1))
    data = sliding_window.reshape(sliding_window.shape[0],
                                  sliding_window.shape[2], 
                                  sliding_window.shape[3],1)
    return data


def load_data_features_position(species_list, window, ratio, validation, 
                                features, mode):
    '''
    Load the corresponding data for feature position prediction to give 
    to a training generator. The data will be one hot encoded and loaded 
    in a vectorized sliding window for better performances.
    '''
    data = np.zeros((window//2,4),dtype='int8')
    indexes = np.array([], dtype=int)
    labels = np.array([], dtype='int8')
    total_len = 0

    for species in species_list:
        files = os.listdir(f'Data/DNA/{species}/one_hot')
        annot = pd.read_csv(f'Data/Annotations/{species}/{features}.csv', 
                            sep = ',')
        annot = annot.drop_duplicates(subset=['chr', 'stop', 'start', 'strand'],
                                      keep='last') 
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
    

def load_data_coverage(species_list, window, step, validation, mode, 
                       chr_nb, features):
    '''
    Load the corresponding data for coverage prediction to give to a training 
    generator. The data will be one hot encoded and loaded in a vectorized 
    sliding window for better performances.
    '''
    dna = np.zeros((window//2,4),dtype='int8')
    pred = np.zeros((window//2,mode*2),dtype='float32')
    indexes = np.array([], dtype=int)
    labels = np.array([], dtype='int8')
    total_len = 0

    for species in species_list:
        files = os.listdir(f'Data/DNA/{species}/one_hot')
        annot = pd.read_csv(f'Data/Annotations/{species}/{features}.csv',
                            sep = ',')
        annot = annot.drop_duplicates(subset=['chr', 'stop', 'start', 'strand'],
                                      keep='last')
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

            indexes = np.append(indexes, np.arange(total_len, total_len+len(chr)+1,
                                step= step))
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
                chr_pred = np.reshape(chr_pred.flatten(order='F'), (chr_pred.shape[1],
                                      chr_pred.shape[0]))
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

    