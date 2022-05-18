#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:52 2022
@author: lou

Example of use :
python Model_training_prot.py -s test -m Conv_prot
"""
import numpy as np
import argparse
import os
from ModuleLibrary.generators import Generator_Protein
from ModuleLibrary.metrics import MCC, BA
from ModuleLibrary.models import Model_dic
from ModuleLibrary.callbacks import check_pointer, class_weights, early_stopping
from ModuleLibrary.utils import load_data_prot

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--results',
                        help="Results suffix")
    parser.add_argument('-m', '--model',
                        help="Model architecture to use")
    parser.add_argument('-o', '--mode',
                        help="Mode to use")                    
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help="Number of epochs")
    parser.add_argument('-w', '--window', default=10, type=int,
                        help="Window size")
    parser.add_argument('-b', '--batch_size', default=2048, type=int,
                        help="Batch size")
    parser.add_argument('-v', '--verbose', default=1, type=int,
                        help="Verbose mode")      
    return parser.parse_args()

def main():

    args = parse_arguments()

    print('Loading data')
    data, labels,  train, val = load_data_prot(args.window, 0.15, args.mode)

    model = Model_dic(args.window)[args.model]   

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',   
                  metrics=['accuracy', MCC, BA])

    model.summary() 

    
    
    

    train_generator = Generator_Protein(indexes = train, 
                                        labels = labels,
                                        data = data, 
                                        batch_size = args.batch_size,
                                        window = args.window,
                                        mode = args.mode,
                                        shuffle = True)

    val_generator = Generator_Protein(indexes = val, 
                                      labels = labels,
                                      data = data, 
                                      batch_size = args.batch_size,
                                      window = args.window,
                                      mode = args.mode,
                                      shuffle = True)

    path = f'Results/{args.results}'
    try:
        os.mkdir(path)
    except:
        print("\nOverwriting results\n")

    # Hyperparameters and callbacks
    callbacks = [check_pointer(path),early_stopping()]

    
    # Model training    
    history = model.fit(train_generator,
                        epochs = args.epochs,
                        batch_size =  args.batch_size,
                        validation_data = val_generator,
                        callbacks = callbacks,
                        verbose = args.verbose)

    # Results saving
    print(f'\nSaving results in : {path}')
    model.save(f'{path}/model.hdf5')

    np.save(f'{path}/loss.npy', history.history['loss'])
    np.save(f'{path}/acc.npy', history.history['accuracy'])
    np.save(f'{path}/MCC.npy', history.history['MCC'])
    np.save(f'{path}/BA.npy', history.history['BA'])
    np.save(f'{path}/val_loss.npy', history.history['val_loss'])
    np.save(f'{path}/val_acc.npy', history.history['val_accuracy'])
    np.save(f'{path}/val_MCC.npy', history.history['val_MCC'])
    np.save(f'{path}/val_BA.npy', history.history['val_BA'])

if __name__ == '__main__':
    main()