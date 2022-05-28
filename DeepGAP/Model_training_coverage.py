#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Jan 19 15:52 2022
@author: Lou Duron

Model training for feature coverage on sequences:
    --results, -r, {str} : Result suffix used
    --species, -s, {list(str)} : List of species ID to train on
    --model, -m, {str} : Model to use
    --feature, -f, {str} : Feature to use (GENE, EXON, RNA...etc)
    --chr, -c, {int} : Number of chromosome to use (0 for all)
    --epochs, -e, {int} : Number of epochs
    --window, -w, {int} : Window size
    --step, -t, {int} : Steps between windows
    --batch_size, -b, {int} : Batch size
    --verbose, -v, {int} : Verbose mode

Example of use :
python Model_training_coverage.py -r myResults -s Clin Maca Leca -m Conv -f RNA -c 0
"""

import numpy as np
import argparse
import os
import sys
sys.path.insert(0,'..')
from ModuleLibrary.generators import Generator_Coverage
from ModuleLibrary.metrics import MCC, BA
from ModuleLibrary.models import Model_dic
from ModuleLibrary.callbacks import check_pointer, class_weights, early_stopping
from ModuleLibrary.data_loaders import load_data_coverage

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results', required=True, 
                        help="Results suffix")
    parser.add_argument('-s','--species', nargs='+', required=True,
                        help='Species list',)
    parser.add_argument('-m', '--model', required=True, 
                        help="Model architecture to use")
    parser.add_argument('-f', '--feature', default= 'GENE',
                        type=str, help="Feature to use")
    parser.add_argument('-c', '--chr', default=1, type=int,
                        help="Number of chromosome to train on, 0 for all")
    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help="Number of epochs")
    parser.add_argument('-w', '--window', default=2000, type=int,
                        help="Window size")
    parser.add_argument('-t', '--step', default=100, type=int,
                        help="Step between windows")
    parser.add_argument('-b', '--batch_size', default=1024, type=int,
                        help="Batch size")
    parser.add_argument('-v', '--verbose', default=1, type=int,
                        help="Verbose mode")     
    return parser.parse_args()

def main():

    args = parse_arguments()

    window = args.window
    if window % 2 == 0:
        window += 1
    
    print('Loading data')
    data, labels, ratio, train_indexes, val_indexes  = load_data_coverage(args.species, 
                                                                          window, 
                                                                          args.step, 
                                                                          0.15,
                                                                          args.chr,
                                                                          args.feature)

    train_generator = Generator_Coverage(indexes = train_indexes, 
                                         labels = labels,
                                         data = data, 
                                         batch_size = args.batch_size,
                                         window = window,
                                         shuffle = True,
                                         data_augment=True)

    val_generator = Generator_Coverage(indexes = val_indexes, 
                                       labels = labels,
                                       data = data,
                                       batch_size = args.batch_size,
                                       window = window,
                                       shuffle = True,
                                       data_augment=False)


    model = Model_dic(window)[args.model]   

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',   
                  metrics=['accuracy', MCC, BA])
    model.summary() 

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
                        validation_data=val_generator,
                        callbacks = callbacks,
                        verbose = args.verbose, 
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