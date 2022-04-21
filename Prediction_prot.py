#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Feb 23 14:44 2022
@author: lou

python Prediction_prot.py -p prot5_hs37_30 -s HS37 -c 1 -r test
"""
import numpy as np
import argparse
import os
from tensorflow import keras
from ModuleLibrary.metrics import MCC, BA
from ModuleLibrary.utils import load_data_one_chr_prot
from ModuleLibrary.generators import Generator_Prediction_prot

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix',
                        help="Input prefix")
    parser.add_argument('-s', '--species', type=str,
                        help="Target species")
    parser.add_argument('-c', '--chromosome', type=str,
                        help="Taget chromosome")
    parser.add_argument('-r', '--result', 
                        help="Results prefix")
    parser.add_argument('-v', '--reverse', action='store_true',
                        help="Predict on strand -")                
    return parser.parse_args()

def main():
    args = parse_arguments()

    print('Loading model')

    path = f'Results/{args.prefix}/best_metrics_model.hdf5'
    model = keras.models.load_model(path,
                                    custom_objects={'MCC': MCC,
                                                    'BA' : BA})

    window_size = model.get_layer(index=0).input_shape[0][1]

    print('Loading data')

    data = load_data_one_chr_prot(args.species, args.chromosome, window_size,
                                        args.reverse)

    pred_generator = Generator_Prediction_prot(data = data, 
                                          batch_size = 2048,
                                          window = window_size)

    prediction = model.predict(pred_generator, verbose=1)

    if args.reverse:
        prediction = prediction[::-1]

    np.save(f'Predictions/{args.result}.npy', prediction)         
    
if __name__ == '__main__':
    main()