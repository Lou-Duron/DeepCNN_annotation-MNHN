#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Feb 15 2022
@author: Lou Duron

This module contains plotting function for training and 
prediction analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py

def training_metrics(prefix):
    '''
    Metrics evolution plots
    - prefix {str}: forlder name with training results
    '''
    plt.style.use('seaborn-white')

    hist_loss = np.load(f'Results/{prefix}/loss.npy')
    hist_acc = np.load(f'Results/{prefix}/acc.npy')
    hist_MCC = np.load(f'Results/{prefix}/MCC.npy')
    hist_BA = np.load(f'Results/{prefix}/BA.npy')

    hist_val_loss = np.load(f'Results/{prefix}/val_loss.npy')
    hist_val_acc = np.load(f'Results/{prefix}/val_acc.npy')
    hist_val_MCC = np.load(f'Results/{prefix}/val_MCC.npy')
    hist_val_BA = np.load(f'Results/{prefix}/val_BA.npy')   

    epochs = np.arange(1, len(hist_loss) + 1 )

    figure, axis = plt.subplots(2, 2, constrained_layout=True,
                                figsize=(8.5,5.5))
    
    axis[0, 0].plot(epochs, hist_loss, label ='Training')
    axis[0, 0].plot(epochs, hist_val_loss, label ='Validation')
    axis[0, 0].set_title("Loss")

    axis[0, 1].plot(epochs, hist_acc, label ='Training')
    axis[0, 1].plot(epochs, hist_val_acc, label ='Validation')
    axis[0, 1].set_title("Accuracy")

    axis[1, 0].plot(epochs, hist_BA, label ='Training')
    axis[1, 0].plot(epochs, hist_val_BA, label ='Validation')
    axis[1, 0].set_title("Balanced accuracy")

    axis[1, 1].plot(epochs, hist_MCC, label ='Training')
    axis[1, 1].plot(epochs, hist_val_MCC, label ='Validation')
    axis[1, 1].set_title("MCC")
    axis[1, 1].legend()

    plt.show()

def prediction_density(pred_files, species, chr, annotation, win_range,
                       mode='all', annot_end=False):
    '''
    Plots prediction mean around target features
    - pred_files {list(str)}: list of target file names
    - species {str}: species name
    - chr {str/int}: chromosome number
    - annotation {str}: target features (GENE, CDS, EXON or RNA) 
    - win_range {int}: range around target feature
    - mode {str}: genomic strand to use (all, strand+ or strand-)
    - annot_end {bool}: if True will check end of feature (example: Gene end)
    '''
    annot = pd.read_csv(f'Data/Annotations/{species}/{annotation}.csv',
                        sep = ',')
    annot = annot.drop_duplicates(subset=['chr', 'start', 'stop', 'strand'],
                                  keep='last') 
    annot = annot[(annot.chr == f'chr{chr}' )] 
    annot_5 = annot[(annot.strand == '+')]
    annot_3 = annot[(annot.strand == '-')]
    if annot_end:
        annot_5_index = annot_5['stop'].values
        annot_3_index = annot_3['start'].values
    else:
        annot_5_index = annot_5['start'].values
        annot_3_index = annot_3['stop'].values
    annot_5_index = annot_5_index -1
    annot_3_index = annot_3_index -1
        
    figure, axis = plt.subplots(1, len(pred_files), constrained_layout=True,
                                figsize=(len(pred_files)*5,4))
    for num,pred_file in enumerate(pred_files):
        pred = np.load(f'Predictions/{pred_file}')
        pred = pred.reshape(pred.shape[0])
        name = pred_file.replace('.npy','')
        res_all, res_5, res_3 = [], [], []
        x = np.arange(-(win_range//2),(win_range//2)+1)
        for i in x:
            y_all, y_5, y_3, nb_all, nb_5, nb_3 = 0, 0, 0, 0, 0, 0
            for el in annot_5_index:
                    y_all += pred[el + i]
                    y_5 += pred[el + i]
                    nb_all += 1
                    nb_5 += 1
            for el in annot_3_index:
                    y_all += pred[el + i]
                    y_3 += pred[el + i]
                    nb_all += 1
                    nb_3 += 1
            res_all.append(y_all/nb_all)
            res_5.append(y_5/nb_5)
            res_3.append(y_3/nb_3)
        if len(pred_files) > 1:
            plot = axis[num]
        else:
            plot = plt
            plot.title(name)
        if mode == 'all' or mode == 'strand+':
            plot.plot(x,res_5, label='Strand +')
        if mode == 'all' or mode == 'strand-':
            plot.plot(x,res_3, label='Strand -')
        if mode == 'all':
            plot.plot(x,res_all, label='All')
        if len(pred_files) > 1:
            plot.set_title(name)
        plot.legend()
    plt.show

def prediction_quality(pred_files, species, chr, mode='all'):
    '''
    Plots of the prediction quality for coverage
    - pred_files {list(str)}: list of target file names
    - species {str}: species name
    - chr {str/int}: chromosome number
    - mode {str}: genomic strand to use (all, gene_strand+, exon_strand-...etc)
    '''
    positions = np.load(f'Data/Positions/{species}/{mode}/chr{chr}.npy')
            
    figure, axis = plt.subplots(1, len(pred_files), constrained_layout=True, 
                                figsize=(len(pred_files)*5,4))
    for num,pred_file in enumerate(pred_files):
        pred = np.load(f'Predictions/{pred_file}')
        pred = pred.reshape(pred.shape[0])
        name = pred_file.replace('.npy','')

        positions = positions[:len(pred)]

        pos = np.where(positions==1)[0]
        neg = np.where(positions==0)[0]

        pred_pos = pred[pos]
        pred_neg = pred[neg]
        x = [pred_pos, pred_neg]
        labels = [f'Inside', f'Outside']

        if len(pred_files) > 1:
            plot = axis[num]
        else:
            plot = plt
            plot.title(name)

        plot.boxplot(x, labels=labels, showmeans=True, meanline=True, 
                     widths=0.6, showfliers=False)
        if len(pred_files) > 1:
            plot.set_title(name)
    plt.show

class Chromosome_prediction():
    '''
    Class to explore prediction values along a chromosome
    '''
    def __init__(self, files_list, species, chr_nb, GC,
                 window, conv, annotation_type, mode = 'all'):
        plt.style.use('seaborn-white')
        self.files_list = files_list
        self.species = species
        self.chr_nb = chr_nb
        self.window = window
        self.GC = GC
        self.mode = mode
        self.annot = annotation_type
        f = h5py.File(f'Data/DNA/{species}/hdf5/chr{chr_nb}.hdf5','r')
        DNA = np.array(f['data'])
        f.close()
        DNA = DNA.reshape(DNA.shape[0],)
        self.DNA = DNA.astype('int8')
        self.DNA_size = len(DNA)
        self.update_conv(conv)
        self.update_annotation(annotation_type)
        
    def update_conv(self, conv):
        self.preds = []
        for prediction_file in self.files_list:
            pred = np.load(f'Predictions/{prediction_file}')
            pred = pred.reshape(pred.shape[0],)
            if conv != 0:
                pred = np.convolve(pred, np.ones(conv), 'valid') / conv
                self.GC_cont = np.convolve((self.DNA>=3).astype('int8'), 
                                            np.ones(conv), 'valid') / conv
            self.preds.append(pred)

    def update_annotation(self, annotation_type):
        annot = pd.read_csv(f'Data/Annotations/{self.species}/{annotation_type}.csv',
                            sep = ',')
        annot = annot.drop_duplicates(subset=['chr','stop', 'start', 'strand'],
                                      keep='last') 
        annot = annot[(annot.chr == f'chr{self.chr_nb}' )] 
        annot_5 = annot[(annot.strand == '+')]
        annot_3 = annot[(annot.strand == '-')]
        annot_5_start_index = annot_5['start'].values - 1
        annot_5_stop_index = annot_5['stop'].values - 1
        annot_3_start_index = annot_3['start'].values - 1
        annot_3_stop_index = annot_3['stop'].values - 1
        if self.annot != 'RNA':
            pos_5 = np.zeros(self.DNA_size, dtype='int8')
            pos_3 = np.zeros(self.DNA_size, dtype='int8')

            def fill_5(x):
                pos_5[x] = 1
            fill_5_vec = np.frompyfunc(fill_5, 1,0)

            def fill_3(x):
                pos_3[x] = 1
            fill_3_vec = np.frompyfunc(fill_3, 1,0)

            
            def get_indexes(x,y):
                    return np.arange(x,y+1)
            get_indexes_vec = np.frompyfunc(get_indexes,2,1)

        
            index_5_pos = get_indexes_vec(annot_5_start_index,annot_5_stop_index)
            index_5_pos = np.concatenate(index_5_pos)
            index_5_pos = np.unique(index_5_pos)

            index_3_pos = get_indexes_vec(annot_3_start_index,annot_3_stop_index)
            index_3_pos = np.concatenate(index_3_pos)
            index_3_pos = np.unique(index_3_pos)

            fill_5_vec(index_5_pos)
            fill_3_vec(index_3_pos)
        

            self.pos_5 = pos_5
            self.pos_3 = pos_3*0.98
            self.nuc = np.arange(self.DNA_size)
        else:
            pos_5 = []
            for el in range(5):
                pos_5.append(np.zeros(self.DNA_size, dtype='int8'))
            pos_3 = []
            for el in range(5):
                pos_3.append(np.zeros(self.DNA_size, dtype='int8'))

            def fill_5(x,i):
                pos_5[i][x] = 1
            fill_5_vec = np.frompyfunc(fill_5, 2,0)

            def fill_3(x,i):
                pos_3[i][x] = 1
            fill_3_vec = np.frompyfunc(fill_3, 2,0)

            def get_indexes(x,y):
                    return np.arange(x,y+1, dtype=int)
            get_indexes_vec = np.frompyfunc(get_indexes,2,1)

            index_5_pos = get_indexes_vec(annot_5_start_index,annot_5_stop_index)
            for i in range(len(index_5_pos)):
                fill_5_vec(index_5_pos[i], i%5)
        
            index_3_pos = get_indexes_vec(annot_3_start_index,annot_3_stop_index)
            for i in range(len(index_3_pos)):
                fill_3_vec(index_3_pos[i], i%5)

            for i in range(5):
                pos_5[i] = pos_5[i] * (0.9 + (i/50))
                pos_3[i] = pos_3[i] * (0.9 + (i/50))

            self.pos_5 = pos_5
            self.pos_3 = pos_3
            self.nuc = np.arange(self.DNA_size)

    def update_plots(self, plot_start, plot_range):
        x = plot_start
        y = plot_range
        plt.figure(figsize=(30,4))
        plt.ylim([0,1.1])
        if self.mode == 'all':
            plt.plot(self.nuc[x:x+y], self.pos_5[x:x+y], label='Strand +')
            plt.plot(self.nuc[x:x+y], self.pos_3[x:x+y], label='Strand -')
        elif self.mode == 'strand+':
            if self.annot != 'RNA':
                plt.plot(self.nuc[x:x+y], self.pos_5[x:x+y], label='Strand +')
            else:
                for i in range(5):
                    plt.plot(self.nuc[x:x+y], self.pos_5[i][x:x+y])
        elif self.mode == 'strand-':
            plt.plot(self.nuc[x:x+y], self.pos_3[x:x+y], label='Strand -')
        if self.GC:
            plt.plot(self.nuc[x:x+y], self.GC_cont[x:x+y], label='GC%')
        for i, pred in enumerate(self.preds):
            name = self.files_list[i]
            name = name.replace('.npy','')
            plt.plot(self.nuc[x:x+y], pred[x:x+y], label=name)
        plt.legend()
        plt.show()
