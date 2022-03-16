#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 11:21 2022
@author: lou

Example of use :
python Data_generator.py -a GENE -w 300 -r 100 -f multi2
"""

import numpy as np
import pandas as pd
import argparse
import h5py
import os
import pickle
from ModuleLibrary.utils import sliding_window_view

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--annotation',
                        help="Annotation type")
    parser.add_argument('-w', '--window', default=300, type=int,
                        help='Window size')
    parser.add_argument('-r', '--ratio', default=100, type=int,
                        help='Ratio postive/Negative samples')                   
    parser.add_argument('-f', '--prefix',
                        help="Results prefix")
    parser.add_argument('-p', '--stop', action='store_true',
                        help="Get end of genes/CDSs, start if not")
            
    return parser.parse_args()

def get_species_list():
    list = [#'Maca', # 88 artefacts
            'HS37', # 6 artefacts
            #'Call', # 11 artefacts
            #'LeCa', # 1 Artefact
            #'PanP', # 34 artefacts
            #'Asia', # 13 artefacts + delete
            #'ASM2', #  416 artefacts + delete
            #'ASM7', # 32 artefacts
            #'Clin', # 26 artefacts <--- Pred
            #'Kami', # 49 artefacts
            #'Mmul', # 4 artefacts
            #'Panu', # 58 artefacts <--- Pred2
            #'Tgel' # 212 artefacts
            #'Cani',
            #'Dani',
            #'Equi',
            #'Feli',
            #'Gall',
            #'MusM',
            #'Orni'
            ] #multi3 = hs37, call, Leca, PanP, Asia, ASM7, Clin, Mmul

    return list

def main():
    args = parse_arguments()    

    # Resize window to be an odd number 
    window = args.window
    if window % 2 == 0:
        window += 1
    n = window // 2

    species_list = get_species_list()

    dataset_infos ={'labels' : {},
                    'species': species_list,
                    'window' : window,
                    'ratio' : args.ratio}

    for species in species_list:

        print(f'\n{species}')
        # Get annotation dataframe and remove duplicates
        annot = pd.read_csv(f'Data/Annotations/{species}/{args.annotation}.csv', sep = ',')
        annot = annot.drop_duplicates(subset=['chr', 'stop', 'start', 'strand'], keep='last') 

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

            # Split strand specific ANNOTs
            ANNOT_5 = ANNOT[(ANNOT.strand == '+')]
            ANNOT_3 = ANNOT[(ANNOT.strand == '-')]
            
            # Get start or stop ANNOT position
            if args.stop :
                index_pos_5 = ANNOT_5['stop'].values
                index_pos_3 = ANNOT_3['start'].values
            else:
                index_pos_5 = ANNOT_5['start'].values
                index_pos_3 = ANNOT_3['stop'].values

            index_pos_5 = index_pos_5[index_pos_5 > n]
            index_pos_3 = index_pos_3[index_pos_3 > n]
            index_pos_5 = index_pos_5[index_pos_5 < (len(DNA) - n)]
            index_pos_3 = index_pos_3[index_pos_3 < (len(DNA) - n)]

            # Get DNA sequences centered on ANNOT start/stop position
            DNA_win = sliding_window_view(DNA, window)
            ANNOT_5_seq = DNA_win[index_pos_5 - n - 1]
            ANNOT_3_seq = DNA_win[index_pos_3 - n - 1]

            # Create an array containing indexes of all nucleotides in positive sequences
            indexes_to_remove = np.array([], dtype = 'int32')

            for i in index_pos_5:
                range_to_remove = np.arange(i - window, i + window + 1, dtype = 'int32')
                indexes_to_remove = np.append(indexes_to_remove, range_to_remove)

            for i in index_pos_3:
                range_to_remove = np.arange(i - window, i + window + 1, dtype = 'int32')
                indexes_to_remove = np.append(indexes_to_remove, range_to_remove)
                
            # Remove artefacts
            '''
            artefact5 = np.sort(np.unique(np.where(ANNOT_5_seq == 0)[0], return_index = True)[0])[::-1]
            if len(artefact5) > 0 :
                print(f'{len(artefact5)} artefact(s) detected on strand + : removing')
                for art in artefact5:
                    index_pos_5 = np.delete(index_pos_5, art, axis=0)

            artefact3 = np.sort(np.unique(np.where(ANNOT_3_seq == 0)[0], return_index = True)[0])[::-1]
            if len(artefact3) > 0 :
                print(f'{len(artefact3)} artefact(s) detected on strand - : removing')
                artefact3.sort
                for art in artefact3:
                    index_pos_3 = np.delete(index_pos_3, art, axis=0)
            '''
            # Removing positive indexes from potential negative indexes
            index_pool = np.arange(len(DNA))
            indexes_to_remove = indexes_to_remove[indexes_to_remove < len(DNA)-1]
            index_pool = np.delete(index_pool, indexes_to_remove)
            index_pool = index_pool[index_pool > n]
            index_pool = index_pool[index_pool <= len(DNA) - n]

            # Shuffle indexes
            np.random.shuffle(index_pool)

            # Get positive sequences according the ratio
            index_neg = index_pool[:(len(index_pos_5) + len(index_pos_3)) * args.ratio]

            chr_id = chr_id.replace('chr','')
            for index in index_pos_5:
                ID = f'{species}_{chr_id}_{index}_5'
                dataset_infos['labels'][ID] = 1

            for index in index_pos_3:
                ID = f'{species}_{chr_id}_{index}_3'
                dataset_infos['labels'][ID] = 1

            for index in index_neg:
                ID = f'{species}_{chr_id}_{index}_5'
                dataset_infos['labels'][ID] = 0


    path = f'Input/{args.prefix}'

    try:
        os.mkdir(path)
    except:
        print("Overwriting")

    with open(f'{path}/dataset_infos.pkl', 'wb') as f:
            pickle.dump(dataset_infos, f)

if __name__ == '__main__':
    main()