#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:30 2022
@author: lou
"""
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

def class_weights(ratio):
    cw = {0 : ((ratio + 1) / 2) / ratio, 1 : ((ratio + 1) / 2)}
    print(f'Class weights : {cw}')
    return cw


def check_pointer(path):

    checkpointer = ModelCheckpoint(filepath=f'{path}/best_metrics_model.hdf5',
                                monitor='val_MCC',
                                mode='max', 
                                verbose=0, 
                                save_best_only=True, 
                                save_weights_only=False, 
                                period=1)
    return checkpointer

def early_stopping():

    early = EarlyStopping(monitor='val_loss', 
                            min_delta=0, 
                            patience=5, 
                            verbose=0, 
                            mode='min',
                            restore_best_weights=True)
    return early


def batch_history(): 
    class History(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []
            self.MCC = []
            self.BA = []
            self.val_losses = []
            self.val_MCC = []
            self.val_BA = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))
            self.MCC.append(logs.get('MCC'))
            self.BA.append(logs.get('BA'))
            self.val_losses.append(logs.get('val_loss'))
            self.val_MCC.append(logs.get('val_MCC'))
            self.val_BA.append(logs.get('val_BA'))
    
    history = History()

    return history