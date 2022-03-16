#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 1 10:53 2022
@author: lou

Example of use :
python Regression_parser.py
"""

import numpy as np
import os

def get_species_list():
    list = [#'Maca', # 88 artefacts
            'HS37', # 6 artefacts
            #'Call', # 11 artefacts
            #'LeCa', # 1 Artefact
            #'PanP', # 34 artefacts
            #'Asia', # 13 artefacts
            #'ASM2', #  416 artefacts
            #'ASM7', # 32 artefacts
            #'Clin', # 26 artefacts <--- Pred
            #'Kami', # 49 artefacts
            #'Mmul', # 4 artefacts
            #'Panu', # 58 artefacts
            #'Tgel' # 212 artefacts
            ] 
    return list

def main():    

    species_list = get_species_list()

    for species in species_list:

        path = f'Data/Positions/{species}/regression'
        try:
            os.mkdir(path)
        except:
            print("Overwriting")
        chromosomes = os.listdir(f'Data/DNA/{species}/hdf5')

        for chr in chromosomes:
            chr_id = chr.replace('.hdf5','')
            pos = np.load(f'Data/Positions/{species}/full/{chr_id}.npy')
            test = np.convolve(pos, np.ones(300), 'valid')
            test = np.append(np.zeros(150), test)

            np.save(f'{path}/{chr_id}.npy', test)

if __name__ == '__main__':
    main()