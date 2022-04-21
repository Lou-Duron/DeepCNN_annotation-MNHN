#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:55 2022
@author: lou
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow import keras
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, matthews_corrcoef
from ModuleLibrary.metrics import MCC, BA
from ModuleLibrary.utils import sliding_window_view, OH_to_DNA

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
    annot_5_index = annot_5_index -1
    annot_3_index = annot_3_index -1
        
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

def prediction_quality(pred_files, species, chr, mode='all'):

    positions = np.load(f'Data/Positions/{species}/{mode}/chr{chr}.npy')
            
    figure, axis = plt.subplots(1, len(pred_files), constrained_layout=True, figsize=(len(pred_files)*5,4))
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

        plot.boxplot(x, labels=labels, showmeans=True, meanline=True, widths=0.6, showfliers=False)
        if len(pred_files) > 1:
            plot.set_title(name)
    plt.show

class Chromosome_prediction():

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
                self.GC_cont = np.convolve((self.DNA>=3).astype('int8') , np.ones(conv), 'valid') / conv
            self.preds.append(pred)

    def update_annotation(self, annotation_type):
        annot = pd.read_csv(f'Data/Annotations/{self.species}/{annotation_type}.csv', sep = ',')
        annot = annot.drop_duplicates(subset=['chr','stop', 'start', 'strand'], keep='last') 
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


class Chromosome_prediction_prot():

    def __init__(self, files_list, species, chr_nb, annotation_type):

        plt.style.use('seaborn-white')
        self.species = species
        self.chr_nb = chr_nb
        self.annot = annotation_type
        self.file = files_list[0]
        f = h5py.File(f'Data/DNA/{species}/hdf5/chr{chr_nb}.hdf5','r')
        DNA = np.array(f['data'])
        f.close()
        DNA = DNA.reshape(DNA.shape[0],)
        self.DNA = DNA.astype('int8')
        self.DNA_size = len(DNA)
        pred = np.load(f'Predictions/{self.file}')
        self.pred = pred.reshape(pred.shape[0],)
        self.update_annotation(annotation_type)
        self.update_conv(10)
        
    def update_conv(self, conv):
        self.frames = []
        for i in range(3):
            self.frames.append(np.copy(self.pred))
        for i in range(len(self.pred)):
            if i % 3 != 0:
                self.frames[0][i] = 0
        for i in range(len(self.pred)):
            if i % 3 != 1:
                self.frames[1][i] = 0
        for i in range(len(self.pred)):
            if i % 3 != 2:
                self.frames[2][i] = 0
        for i in range(3):
            self.frames[i] = (np.convolve(self.frames[i], np.ones(conv), 'valid') / conv)*3

    def update_annotation(self, annotation_type):
        self.annot = annotation_type
        annot = pd.read_csv(f'Data/Annotations/{self.species}/{annotation_type}.csv', sep = ',')
        annot = annot.drop_duplicates(subset=['chr','stop', 'start', 'strand'], keep='last') 
        annot = annot[(annot.chr == f'chr{self.chr_nb}' )] 
        annot_5 = annot[(annot.strand == '+')]
        annot_3 = annot[(annot.strand == '-')]
        annot_5_start_index = annot_5['start'].values - 1
        annot_5_stop_index = annot_5['stop'].values - 1
        annot_3_start_index = annot_3['start'].values - 1
        annot_3_stop_index = annot_3['stop'].values - 1
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
       

    def update_plots(self, plot_start, plot_range):
        x = plot_start
        y = plot_range
        plt.figure(figsize=(30,4))
        plt.ylim([0,1.1])
        
        plt.plot(self.nuc[x:x+y], self.pos_5[x:x+y], label=self.annot)
        plt.plot(self.nuc[x:x+y], self.frames[0][x:x+y], label='Frame 1')
        plt.plot(self.nuc[x:x+y], self.frames[1][x:x+y], label='Frame 2')
        plt.plot(self.nuc[x:x+y], self.frames[2][x:x+y], label='Frame 3')
        plt.legend()
        plt.show()

class Features_exploration():

    def __init__(self, model_path, pred_path, species, chromosome, treshold, motif):
        self.model = keras.models.load_model(model_path, custom_objects={'MCC': MCC, 'BA' : BA})
        self.model_w = h5py.File(model_path,'r')
        self.pred = np.load(pred_path)
        self.chr = np.load(f'Data/DNA/{species}/one_hot/chr{chromosome}.npy').astype('int32')
        self.TRESHOLD = treshold
        self.load_weights()
        self.init_motif(motif)
        
        
    def init_motif(self, motif):
        self.motif = []
        if motif == 'exon_start':
            m = np.array(['c','a','g','g','a'])
        elif motif == 'exon_stop':
            m = np.array(['g','g','t','a','a'])
        mut = [0,3,4]
        for i in mut:
            for j in ['a','t','c','g']:
                for k in mut:
                    for l in ['a','t','c','g']:
                        if i != k:
                            tmp = np.copy(m)
                            tmp[i] = j
                            tmp[k] = l
                            self.motif.append(tmp)

    def load_weights(self):
        # Layers
        self.conv1 = Model(inputs=self.model.inputs, outputs=self.model.get_layer(name="conv2d").output) 
        self.mp1 = Model(inputs=self.model.inputs, outputs=self.model.get_layer(name="max_pooling2d").output)
        self.conv2 = Model(inputs=self.model.inputs, outputs=self.model.get_layer(name="conv2d_1").output) 
        self.mp2 = Model(inputs=self.model.inputs, outputs=self.model.get_layer(name="max_pooling2d_1").output)
        self.conv3 = Model(inputs=self.model.inputs, outputs=self.model.get_layer(name="conv2d_2").output) 
        self.mp3 = Model(inputs=self.model.inputs, outputs=self.model.get_layer(name="max_pooling2d_2").output)
        self.flatten = Model(inputs=self.model.inputs, outputs=self.model.get_layer(name="flatten").output)
        self.d1 = Model(inputs=self.model.inputs, outputs=self.model.get_layer(name="dense").output)
        self.d2 = Model(inputs=self.model.inputs, outputs=self.model.get_layer(name="dense_1").output)
        self.dr = Model(inputs=self.model.inputs, outputs=self.model.get_layer(name="dropout_2").output)

        # Layers weights
        self.conv1_w = np.reshape(np.array(self.model_w['model_weights']['conv2d']['conv2d']['kernel:0']), (6,4,32)) 
        self.conv2_w = np.reshape(np.array(self.model_w['model_weights']['conv2d_1']['conv2d_1']['kernel:0']), (6,32,64)) 
        self.conv3_w = np.reshape(np.array(self.model_w['model_weights']['conv2d_2']['conv2d_2']['kernel:0']), (6,64,128))
        self.d1_w = np.array(self.model_w['model_weights']['dense']['dense']['kernel:0']) # (4736,128)
        self.d2_w = np.reshape(np.array(self.model_w['model_weights']['dense_1']['dense_1']['kernel:0']), (128)) 

        # Layers bias
        self.conv1_b = np.array(self.model_w['model_weights']['conv2d']['conv2d']['bias:0']) # (32)
        self.conv2_b = np.array(self.model_w['model_weights']['conv2d_1']['conv2d_1']['bias:0']) # (64)
        self.conv3_b = np.array(self.model_w['model_weights']['conv2d_2']['conv2d_2']['bias:0']) # (128)
        self.d1_b = np.array(self.model_w['model_weights']['dense']['dense']['bias:0']) # (128)
        self.d2_b = np.array(self.model_w['model_weights']['dense_1']['dense_1']['bias:0']) # (1)

    def explore(self, index):
        data_oh = self.chr[index-150:index+151]
        data = data_oh.reshape(1,data_oh.shape[0], data_oh.shape[1],1)

        # Layers output
        conv1_output = np.reshape(self.conv1.predict(data), (296,32))
        mp1_output = np.reshape(self.mp1.predict(data), (148,32))
        conv2_output = np.reshape(self.conv2.predict(data), (148,64))
        mp2_output = np.reshape(self.mp2.predict(data), (74,64))
        conv3_output = np.reshape(self.conv3.predict(data), (74,128))
        mp3_output = np.reshape(self.mp3.predict(data), (37,128))
        flatten_output = np.reshape(self.flatten.predict(data), (4736))
        d1_output = np.reshape(self.d1.predict(data), (128))
        d2_output = np.reshape(self.d2.predict(data), (1))

        print(f'Prediction : {round(d2_output[0],3)}')

        # Dense 2 
        d1_output_contrib = np.zeros(d1_output.shape)
        for i in range(d1_output.shape[0]):
            d1_output_contrib[i] = d1_output[i] * self.d2_w[i]
        
        # Dense 1
        flatten_output_contrib = np.zeros(flatten_output.shape)
        for i in range(flatten_output.shape[0]): # 4736
            for j in range(self.d1_w.shape[1]): # 128
                x = flatten_output[i] * self.d1_w[i][j] * d1_output_contrib[j]
                if d1_output[j] > 0:
                    flatten_output_contrib[i] += x
                else:
                    flatten_output_contrib[i] += -x

        # Unflatten

        mp3_output_contrib = flatten_output_contrib.reshape((37,128))

        # Unmaxpooling 1 

        conv3_output_contrib = np.zeros((74,128))
        for i in range(0,74,2):
            for j in range(128):
                if conv3_output[i][j] > conv3_output[i+1][j]:
                    conv3_output_contrib[i][j] = mp3_output_contrib[int(i/2)][j]
                else:
                    conv3_output_contrib[i+1][j] = mp3_output_contrib[int(i/2)][j]

        
        # Conv3
        mp2_output_contrib = np.zeros((74,64))
        mp2_output_reshaped = padding_slidding(mp2_output,6)

        for i, eli in enumerate(mp2_output_reshaped): # 74
            for l in range(128): # 128
                for j, elj in enumerate(eli): # 6
                    for k, elk in enumerate(elj): # 64
                        x = elk * self.conv3_w[j][k][l] * conv3_output_contrib[i][l]
                        if conv3_output[i][l] > 0:
                            mp2_output_contrib[i][k] += x
                        else:
                            mp2_output_contrib[i][k] += -x
        
        # Unmaxpooling 2 
        conv2_output_contrib = np.zeros((148,64))

        for i in range(0,148,2):
            for j in range(64):
                if conv2_output[i][j] > conv2_output[i+1][j]:
                    conv2_output_contrib[i][j] = mp2_output_contrib[int(i/2)][j]
                else:
                    conv2_output_contrib[i+1][j] = mp2_output_contrib[int(i/2)][j]
        
        # Conv2
        mp1_output_contrib = np.zeros((148,32))
        mp1_output_reshaped = padding_slidding(mp1_output,6)

        for i, eli in enumerate(mp1_output_reshaped): # 148
            for l in range(64): # 64
                for j, elj in enumerate(eli): # 6
                    for k, elk in enumerate(elj): # 32
                        x = elk * self.conv2_w[j][k][l] * conv2_output_contrib[i][l]
                        if conv2_output[i][l] > 0:
                            mp1_output_contrib[i][k] += x
                        else:
                            mp1_output_contrib[i][k] += -x
        # Unmaxpooling 3 
        conv1_output_contrib = np.zeros((296,32))

        for i in range(0,296,2):
            for j in range(32):
                if conv1_output[i][j] > conv1_output[i+1][j]:
                    conv1_output_contrib[i][j] = mp1_output_contrib[int(i/2)][j]
                else:
                    conv1_output_contrib[i+1][j] = mp1_output_contrib[int(i/2)][j]
        
        # Conv1
        seq = data.reshape(data.shape[1], data.shape[2])
        seq = sliding_window_view(seq, 6, axis=0)
        seq = seq.flatten(order='K').reshape((296,6,4))

        seq_contrib = np.zeros((301,4))

        for i, eli in enumerate(seq): # 296
            for l in range(32): # 32
                for j, elj in enumerate(eli): # 6
                    for k, elk in enumerate(elj): # 4
                        x = elk * self.conv1_w[j][k][l] * conv1_output_contrib[i][l]
                        if conv1_output[i][l] > 0:
                            seq_contrib[i+j][k] += x
                        else:
                            seq_contrib[i+j][k] += -x

        maxvalue = max(float(np.max(seq_contrib)),float(-np.min(seq_contrib)))
        for i in range(len(seq_contrib)):
            for j in range(4):
                seq_contrib[i][j] = seq_contrib[i][j] / maxvalue

        DNA = OH_to_DNA(data_oh)


        # Motif search
        DNA_slide = sliding_window_view(DNA, 5)       
        motif_search = {}
        for i, el in enumerate(DNA_slide):
            for mot in self.motif:
                if np.array_equal(el, mot):
                    if i not in motif_search.keys():
                        motif_search[i] = mot

        for el in motif_search.keys():
            contrib = np.sum(seq_contrib[el:el+5])
            if contrib > 0.5:
                mot = ''.join(motif_search[el])
                print(f'Motif found : {mot.upper()} at {el}. Contrib : {round(contrib,3)}')
        
        seq_contrib = np.transpose(seq_contrib)

        cim = plt.imread("ModuleLibrary/colorbar.png")
        cim = cim[cim.shape[0]//2, 50:390, :]

        cmap = mcolors.ListedColormap(cim)
        plt.figure(figsize=(34,1), dpi= 200)
        plt.imshow(seq_contrib, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
        plt.xticks(np.arange(0, 301,10))
        plt.yticks([0,1,2,3], ['a','t','g','c'])
        plt.colorbar()
        plt.show()



def padding_slidding(input, kernel):
    arr = np.append(np.zeros(((kernel // 2) + ((kernel % 2) - 1),input.shape[1])), input, axis=0) 
    arr = np.append(arr, np.zeros((kernel//2,input.shape[1])), axis=0)
    arr = sliding_window_view(arr, kernel, axis=0)
    output = arr.flatten(order='K').reshape((input.shape[0],kernel,input.shape[1]))
    return output