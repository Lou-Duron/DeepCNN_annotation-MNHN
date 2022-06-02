#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wen Feb 23 14:44 2022
@author: Lou Duron

Features coverage prediction on chromosome:
    --prefix, -p, {str} : Result prefix to use
    --species, -s, {str} : Species ID to use
    --chromosome, -c, {str} : Chromosome to use
    --reverse, -r : Reverse results for complement reverse sequences
    --mcc, -m : Use best metrics model instead of last

python Prediction_coverage.py -p res_example -s Panu -c 1 -m
"""

import numpy as np
import argparse
import sys
sys.path.insert(0,'..')
from tensorflow import keras
from ModuleLibrary.metrics import MCC, BA
from ModuleLibrary.data_loaders import load_data_one_chr
from ModuleLibrary.generators import Generator_Prediction_Coverage

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix',
                        help="Input prefix")
    parser.add_argument('-s', '--species', type=str,
                        help="Target species")
    parser.add_argument('-c', '--chromosome', type=str,
                        help="Target chromosome")
    parser.add_argument('-r', '--reverse', action='store_true',
                        help="Predict on strand -")
    parser.add_argument('-m', '--mcc', action='store_true',
                        help="Use mcc model")                        
    return parser.parse_args()

def main():
    args = parse_arguments()

    print('Loading model')
    if args.mcc:
        path = f'Results/{args.prefix}/best_metrics_model.hdf5'
    else:
        path = f'Results/{args.prefix}/model.hdf5'

    model = keras.models.load_model(path,
                                    custom_objects={'MCC': MCC,
                                                    'BA' : BA})
  
    window_size = model.get_layer(index=0).input_shape[0][1]

    print('Loading data')
    data = load_data_one_chr(args.species, args.chromosome, window_size)


    generator = Generator_Prediction_Coverage(data = data, 
                                              batch_size = 1024,
                                              window = window_size)

    prediction = model.predict(generator, verbose=1)

    if args.reverse:
        prediction = prediction[::-1]
        path = f'Predictions/{args.prefix}_{args.species}_chr{args.chromosome}_reverse_pred.npy'
    else:
        path = f'Predictions/{args.prefix}_{args.species}_chr{args.chromosome}_pred.npy'
  
    np.save(path , prediction)         
    
    
if __name__ == '__main__':
    main()