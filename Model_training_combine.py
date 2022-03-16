#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:52 2022
@author: lou

Example of use :
python Model_training_combine.py -m myModel1 -a -r combine_test
"""
import numpy as np
import argparse
import os
from ModuleLibrary.generator import DataGeneratorFull
from ModuleLibrary.metrics import MCC, BA
from ModuleLibrary.models import Model_dic
from ModuleLibrary.hyperparameters import check_pointer, class_weights, early_stopping
from ModuleLibrary.utils import load_data_full
from tensorflow.keras.mixed_precision import experimental as mixed_precision

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results',
                        help="Results suffix")
    parser.add_argument('-m', '--model',
                        help="Model architecture to use")
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help="Number of epochs")
    parser.add_argument('-b', '--batch_size', default=1024, type=int,
                        help="Batch size")
    parser.add_argument('-v', '--validation', default=0.15, type=float,
                        help="Validation ratio")                    
    parser.add_argument('-k', '--kernel', default=6, type=int,
                        help="Kernel size of the convolution layers")
    parser.add_argument('-a', '--early', action='store_true', 
                        help="Use early stopping")          
    return parser.parse_args()


def main():
    args = parse_arguments()
    path = f'Results/{args.results}'

    

    window = 1000
    step = 30


    if window % 2 == 0:
        window += 1

    print('Loading data')
    data = load_data_full(window)


    labels = np.load('Data/Positions/Panu/strand+/chr1_strand+.npy').astype('int8')

    IDs =np.arange(len(data), step = step)
    train_IDs = IDs[int(len(IDs)*args.validation):]
    val_IDs = IDs[:int(len(IDs)*args.validation)]


    IDs_labels = labels[IDs]
    ratio = round(len(IDs_labels[IDs_labels==0])/len(IDs_labels[IDs_labels==1]), 4)

    print(ratio)

    train_generator = DataGeneratorFull(labels = labels,
                                        indexes = train_IDs, 
                                        data = data, 
                                        batch_size = args.batch_size,
                                        window = window,
                                        shuffle = True)

    validation_generator = DataGeneratorFull(labels =  labels,
                                             indexes = val_IDs,
                                             data = data, 
                                             batch_size = args.batch_size,
                                             window = window,
                                             shuffle = True)

    try:
        os.mkdir(path)
    except:
        print("\nOverwriting results\n")

    # Model architecture
    model = Model_dic(window, args.kernel)[args.model]   

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',   
                  metrics=['accuracy', MCC, BA])

    # Hyperparameters and callbacks
    callbacks = [check_pointer(path)]

    if args.early:
        callbacks.append(early_stopping())

    model.summary() 

    # Model training    
    history = model.fit(train_generator,
                        epochs = args.epochs,
                        batch_size =  args.batch_size,
                        validation_data=validation_generator,
                        callbacks = callbacks,
                        verbose = 1,
                        class_weight = class_weights(ratio))

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