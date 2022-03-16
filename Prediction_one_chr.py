#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Feb 23 14:44 2022
@author: lou

python Prediction_one_chr.py -p multi3_RNA_start -s Panu -c 10 -r test
"""
import numpy as np
import argparse
from tensorflow import keras
from ModuleLibrary.metrics import MCC, BA
from ModuleLibrary.utils import load_data_one_chr
from ModuleLibrary.generator import PredGenerator

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
    parser.add_argument('-d', '--dilated', action='store_true',
                        help="Dilated Model architecture")                      
    return parser.parse_args()

def main():
    args = parse_arguments()

    print('Loading model')
    model = keras.models.load_model(f'Results/{args.prefix}/model.hdf5',
                                    custom_objects={'MCC': MCC,
                                                    'BA' : BA})
    print()

    if args.dilated:
        window_size = model.get_layer(index=0).input_shape[0][1]
    else:
        window_size = model.get_layer(index=0).input_shape[1]

    print('Loading data')

    data = load_data_one_chr(args.species, args.chromosome, window_size,
                             args.reverse, padding=True)

    pred_generator = PredGenerator(data = data, 
                                   batch_size = 2048,
                                   window = window_size)

    prediction = model.predict(pred_generator, verbose=1)

    if args.reverse:
        prediction = prediction[::-1]

    np.save(f'Predictions/{args.result}.npy', prediction)         
    
if __name__ == '__main__':
    main()