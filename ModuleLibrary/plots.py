#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:55 2022
@author: lou
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from tensorflow.keras.models import load_model
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, matthews_corrcoef
from ModuleLibrary.metrics import MCC, BA

def training_metrics(prefix):

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

    figure, axis = plt.subplots(2, 2, constrained_layout=True, figsize=(8.5,5.5))
    
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

def prediction_metrics(data_prefix, model_prefix):

    plt.style.use('seaborn-white')

    X_test = np.load(f'Input/{data_prefix}/X_test.npy')
    Y_true = np.load(f'Input/{data_prefix}/Y_test.npy')
    Y_true = Y_true.astype(np.float32).reshape((-1,1))

    model = load_model(f'Results/{model_prefix}/model.hdf5',
                        custom_objects={'MCC': MCC,
                                        'BA': BA})

    Y_pred = model.predict(X_test)
    
    treshold = np.arange(0.01, 1, 0.01)
    
    fn = np.array([])
    tn = np.array([])
    ba = np.array([])
    mcc = np.array([])

    for i in treshold:
        Y_pred_tmp = (Y_pred >= i).astype('int8')
        cm = confusion_matrix(Y_true, Y_pred_tmp)
        p = cm[0][0] + cm[0][1]
        n = cm[1][0] + cm[1][1]
        fn = np.append(fn, cm[0][1]/p)
        tn = np.append(tn, cm[1][1]/n)
        ba = np.append(ba, balanced_accuracy_score(Y_true, Y_pred_tmp))
        mcc = np.append(mcc, matthews_corrcoef(Y_true, Y_pred_tmp))
    
    figure, axis = plt.subplots(2, 1, constrained_layout=True, figsize=(8.5,5.5))

    axis[0].plot(treshold, fn, label ='Gene wrongly predicted')
    axis[0].plot(treshold, tn, label ='Gene correctly predicted')
    axis[0].set_title("Coffusion matrix depending on treshold in %")
    axis[0].legend()

    axis[1].plot(treshold, ba, label='Balanced accuracy')
    axis[1].plot(treshold, mcc, label='MCC')
    axis[1].set_title('Metrics depending on treshold')
    axis[1].legend()
    plt.show()

def prediction_density(pred_files, species, chr, annotation, win_range, mode='all', annot_end=False):
    
    annot = pd.read_csv(f'Data/Annotations/{species}/{annotation}.csv', sep = ',')
    annot = annot.drop_duplicates(subset=['chr', 'start', 'stop', 'strand'], keep='last') 
    annot = annot[(annot.chr == f'chr{chr}' )] 
    annot_5 = annot[(annot.strand == '+')]
    annot_3 = annot[(annot.strand == '-')]
    if annot_end:
        annot_5_index = annot_5['stop'].values
        annot_3_index = annot_3['start'].values
    else:
        annot_5_index = annot_5['start'].values
        annot_3_index = annot_3['stop'].values
        annot = np.append(annot_5_index, annot_3_index)
        
    figure, axis = plt.subplots(1, len(pred_files), constrained_layout=True, figsize=(len(pred_files)*5,4))
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

def prediction_quality(pred_files, species, chr, annotation, mode='all'):

    positions = np.load(f'Data/Positions/{species}/{mode}/chr{chr}.npy')
            
    for num,pred_file in enumerate(pred_files):
        pred = np.load(f'Predictions/{pred_file}')
        pred = pred.reshape(pred.shape[0])
        name = pred_file.replace('.npy','')

        positions = positions[:len(pred)]

        pos = np.where(positions==1)[0]
        neg = np.where(positions==0)[0]

        pred_pos = pred[pos]
        pred_neg = pred[neg]

        mean_pos = np.mean(pred_pos)
        mean_neg = np.mean(pred_neg)
        
        fig = plt.figure()
        plt.title(name)
        ax = fig.add_axes([0,0,1,1])
        labels = ['Inside genes', 'Outside genes']
        values = [mean_pos, mean_neg]
        ax.bar(labels, values)
    plt.show

class Chromosome_prediction():

    def __init__(self, files_list, species, chr_nb, GC,
                 window, conv, annotation_type, mode):
        plt.style.use('seaborn-white')
        self.files_list = files_list
        self.species = species
        self.chr_nb = chr_nb
        self.window = window
        self.GC = GC
        f = h5py.File(f'Data/DNA/{species}/hdf5/chr{chr_nb}.hdf5','r')
        DNA = np.array(f['data'])
        f.close()
        DNA = DNA.reshape(DNA.shape[0],)
        self.DNA = DNA.astype('int8')
        self.DNA_size = len(DNA)
        self.update_conv(conv)
        self.update_annotation(annotation_type, mode)
        
    def update_conv(self, conv):
        self.preds = []
        for prediction_file in self.files_list:
            pred = np.load(f'Predictions/{prediction_file}')
            pred = pred.reshape(pred.shape[0],)
            if conv != 0:
                pred = np.convolve(pred, np.ones(conv), 'valid') / conv
                self.GC_cont = np.convolve((self.DNA>=3).astype('int8') , np.ones(conv), 'valid') / conv
            self.preds.append(pred)

    def update_annotation(self, annotation_type, mode):
        annot = pd.read_csv(f'Data/Annotations/{self.species}/{annotation_type}.csv', sep = ',')
        annot = annot.drop_duplicates(subset=['chr','stop', 'start', 'strand'], keep='last') #########################
        annot = annot[(annot.chr == f'chr{self.chr_nb}' )] 
        annot_5 = annot[(annot.strand == '+')]
        annot_3 = annot[(annot.strand == '-')]
        annot_5_start_index = annot_5['start'].values
        annot_5_stop_index = annot_5['stop'].values
        annot_3_start_index = annot_3['start'].values
        annot_3_stop_index = annot_3['stop'].values

        pos_5 = np.zeros(self.DNA_size, dtype='int8')
        pos_3 = np.zeros(self.DNA_size, dtype='int8')

        def fill_5(x):
            pos_5[x] = 1
        fill_5_vec = np.frompyfunc(fill_5, 1,0)

        def fill_3(x):
            pos_3[x] = 1
        fill_3_vec = np.frompyfunc(fill_3, 1,0)

        if mode == 'full':
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
        elif mode == 'start':
            fill_5_vec(annot_5_start_index)
            fill_3_vec(annot_3_stop_index)
        elif mode == 'stop':
            fill_5_vec(annot_5_stop_index)
            fill_3_vec(annot_3_start_index)

        self.pos_5 = pos_5
        self.pos_3 = pos_3*0.9
        self.nuc = np.arange(self.DNA_size)

    def update_plots(self, plot_start, plot_range):
        x = plot_start
        y = plot_range
        plt.figure(figsize=(30,4))
        plt.ylim([0,1.1])
        plt.plot(self.nuc[x:x+y], self.pos_5[x:x+y], label='Strand +')
        plt.plot(self.nuc[x:x+y], self.pos_3[x:x+y], label='Strand -')
        if self.GC:
            plt.plot(self.nuc[x:x+y], self.GC_cont[x:x+y], label='GC%')
        for i, pred in enumerate(self.preds):
            name = self.files_list[i]
            name = name.replace('.npy','')
            plt.plot(self.nuc[x:x+y], pred[x:x+y], label=name)
        plt.legend()
        plt.show()

