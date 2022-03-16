#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:18 2022
@author: lou

Example of use :
python Data_treatment/one_hot_encode_DNA.py -p HomoSapiens37
"""
import argparse
import os
import h5py
import numpy as np
from ModuleLibrary.utils import one_hot_encoding_seq

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix',
                        help="Directory containing the gff files")
    return parser.parse_args()

def main():
    args = parse_arguments()

    files = os.listdir(f'Data/DNA/{args.prefix}/hdf5')

    try:
        os.mkdir(f'Data/DNA/{args.prefix}/one_hot')
    except:
        print("Overwriting")

    for file in files:
        f = h5py.File(f'Data/DNA/{args.prefix}/hdf5/{file}','r')
        DNA = np.array(f['data'])
        f.close()
        DNA = DNA.reshape(DNA.shape[0],)
        DNA = DNA.astype('int8')

        OH = one_hot_encoding_seq(DNA)

        name = file.replace('.hdf5','')

        np.save(f'Data/DNA/{args.prefix}/one_hot/{name}.npy', OH)

if __name__ == '__main__':
    main()