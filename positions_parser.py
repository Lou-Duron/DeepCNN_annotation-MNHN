#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 11:21 2022
@author: lou

Example of use :
python positions_parser.py -a GENE
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
    return parser.parse_args()  

def get_species_list():
    list = [#'Maca', # 88 artefacts
            #'HS37', # 6 artefacts
            #'Call', # 11 artefacts
            #'LeCa', # 1 Artefact
            #'PanP', # 34 artefacts
            #'Asia', # 13 artefacts
            #'ASM2', #  416 artefacts
            'ASM7', # 32 artefacts
            'Clin', # 26 artefacts <--- Pred
            'Kami', # 49 artefacts
            'Mmul', # 4 artefacts
            'Panu', # 58 artefacts
            'Tgel' # 212 artefacts
            ] 
    return list

def main():
    args = parse_arguments()    

    species_list = get_species_list()

    for species in species_list:

        path = f'Data/Positions/{species}/gene_full'
        try:
            os.mkdir(path)
        except:
            print("Overwriting")

        print(f'\n{species}')
        # Get annotation dataframe and remove duplicates
        annot = pd.read_csv(f'Data/Annotations/{species}/{args.annotation}.csv', sep = ',')
        annot = annot.drop_duplicates(subset=['chr', 'start','stop', 'strand'], keep='last') 

        # Get list of chromosome
        chromosomes = os.listdir(f'Data/DNA/{species}/hdf5')

        # For each chromosome
        for num, chr in enumerate(chromosomes):

            # Get chromosome specific annotation
            chr_id = chr.replace('.hdf5','')
            
            print(f"Chromosome : {chr_id} ({num+1}/{len(chromosomes)})")

            # Get chromosome specific DNA sequence
            f = h5py.File(f'Data/DNA/{species}/hdf5/{chr}','r')
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

            index_pos = np.append(index_pos5, index_pos3)
            index_pos = np.unique(index_pos)
            

            index_full = np.zeros(len(DNA), dtype=bool)
            def fill(x):
                index_full[x-1] = 1
            fill_vec = np.frompyfunc(fill, 1,0)
            fill_vec(index_pos)

            np.save(f'{path}/{chr_id}.npy', index_full)

if __name__ == '__main__':
    main()