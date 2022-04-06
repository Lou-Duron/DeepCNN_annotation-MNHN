#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wen Feb 23 14:44 2022
@author: lou

python Prediction_features.py -p multi3_RNA_start -s Panu -r test
"""
import numpy as np
import argparse
import os
from tensorflow import keras
from ModuleLibrary.metrics import MCC, BA
from ModuleLibrary.utils import load_data_one_chr_coverage
from ModuleLibrary.generators import Generator_Prediction_Coverage

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

    if args.dilated:
        window_size = model.get_layer(index=0).input_shape[0][1]
    else:
        window_size = model.get_layer(index=0).input_shape[1]

    print('Loading data')

    if args.chromosome is not None:

        data = load_data_one_chr_coverage(args.species, args.chromosome, window_size)
        #######
        dna = data[0][:10000000]
        pred = data[1][:10000000]
        data = [dna,pred]
        #######                         

        pred_generator = Generator_Prediction_Coverage(data = data, 
                                                        batch_size = 2048,
                                                        window = window_size)

        prediction = model.predict(pred_generator, verbose=1)


        if args.reverse:
            prediction = prediction[::-1]

        np.save(f'Predictions/{args.result}.npy', prediction)         
    else:

        files = os.listdir(f'Data/DNA/{args.species}/one_hot')
        path = f'Predictions/{args.result}'

        try:
            os.mkdir(path)
        except:
            print("\nOverwriting results\n")

        for file in files:
            chr_id = file.replace('.npy','')
            chr_id = chr_id.replace('chr','')
            data = load_data_one_chr_coverage(args.species, chr_id, window_size,
                                     args.reverse, padding=True)

            pred_generator = Generator_Prediction_Coverage(data = data, 
                                                  batch_size = 2048,
                                                  window = window_size)

            print(f'Predicting chromosome {chr_id}')

            prediction = model.predict(pred_generator, verbose=1)

            assert len(prediction) == len(data), 'Data and prediciton length are not the same'

            if args.reverse:
                prediction = prediction[::-1]

            np.save(f'{path}/chr{chr_id}.npy', prediction)

    
if __name__ == '__main__':
    main()