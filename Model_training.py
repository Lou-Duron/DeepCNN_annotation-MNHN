#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:52 2022
@author: lou

Example of use :
python Model_training_full.py -r example -m myModel1 -e 30 -a -o gene_full
"""
import numpy as np
import argparse
import os
from ModuleLibrary.generators import DataGenerator
from ModuleLibrary.metrics import MCC, BA
from ModuleLibrary.models import Model_dic
from ModuleLibrary.hyperparameters import check_pointer, class_weights, early_stopping
from ModuleLibrary.utils import load_data_multi_species

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results',
                        help="Results suffix")
    parser.add_argument('-o', '--mode',
                        help="Annotation mode")
    parser.add_argument('-m', '--model',
                        help="Model architecture to use")
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help="Number of epochs")
    parser.add_argument('-w', '--window', default=2000, type=int,
                        help="Window size")
    parser.add_argument('-s', '--step', default=100, type=int,
                        help="Step between windows")
    parser.add_argument('-b', '--batch_size', default=1024, type=int,
                        help="Batch size")
    parser.add_argument('-v', '--validation', default=0.15, type=float,
                        help="Validation ratio")                    
    parser.add_argument('-a', '--early', action='store_true', 
                        help="Use early stopping")          
    return parser.parse_args()

def main():

    args = parse_arguments()

    species_list = [#'Maca',
                    'HS37', 
                    #'Call', 
                    #'LeCa',
                    #'PanP', 
                    #'Asia',
                    #'ASM2', 
                    #'ASM7',
                    #'Clin', 
                    #'Kami', 
                    #'Mmul', 
                    #'Panu',
                    #'Tgel', 
                    #'Cani',
                    #'Dani',
                    #'Equi',
                    #'Feli',
                    #'Gall',
                    #'MusM',
                    #'Orni'
                    ]
    
    window = args.window
    if window % 2 == 0:
        window += 1
    
    print('Loading data')
    data, labels, ratio, train_indexes, val_indexes = load_data_multi_species(species_list, 
                                                                              window, 
                                                                              args.step, 
                                                                              args.validation,
                                                                              args.mode)

    train_generator = DataGenerator(indexes = train_indexes, 
                                    labels = labels,
                                    data = data, 
                                    batch_size = args.batch_size,
                                    window = window,
                                    shuffle = True)

    validation_generator = DataGenerator(indexes = val_indexes, 
                                         labels = labels,
                                         data = data, 
                                         batch_size = args.batch_size,
                                         window = window,
                                         shuffle = True)


    # Model 
    model = Model_dic(window)[args.model]   

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',   
                  metrics=['accuracy', MCC, BA])


    path = f'Results/{args.results}'
    try:
        os.mkdir(path)
    except:
        print("\nOverwriting results\n")

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