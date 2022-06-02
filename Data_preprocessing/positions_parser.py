#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Feb 21 11:21 2022
@author: Lou Duron

This program creates numpy arrays indexing features positions
along chromosomes for analysis and plots. The target feature
and strand are given in arguments.

Example of use :
python positions_parser.py -a EXON -s HS38 -t full
"""

import numpy as np
import pandas as pd
import argparse
import h5py
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotation',
                        help="Annotation type") 
    parser.add_argument('-s', '--species',
                        help="Species name")
    parser.add_argument('-t', '--strand', type=str,
                        help="Strand to use : 5, 3 or full")
    return parser.parse_args()  

def main():
    args = parse_arguments()

    path = f'../Data/Positions/{args.species}/{args.annotation}_{args.strand}'
    try:
        os.mkdir(path)
    except:
        print("Overwriting")

    # Get annotation dataframe and remove duplicates
    annot = pd.read_csv(f'../Data/Annotations/{args.species}/{args.annotation}.csv', sep = ',')
    annot = annot.drop_duplicates(subset=['chr', 'start','stop', 'strand'], keep='last') 

    # Get list of chromosome
    chromosomes = os.listdir(f'../Data/DNA/{args.species}/hdf5')

    # For each chromosome
    for num, chr in enumerate(chromosomes):

        # Get chromosome specific annotation
        chr_id = chr.replace('.hdf5','')
        
        print(f"Chromosome : {chr_id} ({num+1}/{len(chromosomes)})")

        # Get chromosome specific DNA sequence
        f = h5py.File(f'../Data/DNA/{args.species}/hdf5/{chr}','r')
        DNA = np.array(f['data'])
        f.close()
        DNA = DNA.reshape(DNA.shape[0],)
        DNA = DNA.astype('int8')
        
        ANNOT = annot[(annot.chr == chr_id )] 

        ANNOT_5 = ANNOT[(ANNOT.strand == '+')]
        ANNOT_3 = ANNOT[(ANNOT.strand == '-')]
        
        # Get start or stop ANNOT position
        ANNOT5_start_index = ANNOT_5['start'].values
        ANNOT5_stop_index = ANNOT_5['stop'].values
        ANNOT3_start_index = ANNOT_3['start'].values
        ANNOT3_stop_index = ANNOT_3['stop'].values

        def get_indexes(x,y):
            return np.arange(x,y+1)

        get_indexes_vec = np.frompyfunc(get_indexes,2,1)

        index_pos5 = get_indexes_vec(ANNOT5_start_index,ANNOT5_stop_index)
        index_pos5 = np.concatenate(index_pos5)
        index_pos5 = np.unique(index_pos5)
        
        index_pos3 = get_indexes_vec(ANNOT3_start_index,ANNOT3_stop_index)
        index_pos3 = np.concatenate(index_pos3)
        index_pos3 = np.unique(index_pos3)

        index_full = np.append(index_pos5, index_pos3)
        index_full = np.unique(index_full)
        

        res = np.zeros(len(DNA), dtype=bool)
        def fill(x):
            res[x-1] = 1
        fill_vec = np.frompyfunc(fill, 1,0)

        if args.strand == '5':
            fill_vec(index_pos5)
        elif args.strand == '3':
            fill_vec(index_pos3)
        elif args.strand == 'full':
            fill_vec(index_full)

        np.save(f'{path}/{chr_id}.npy', res)

if __name__ == '__main__':
    main()