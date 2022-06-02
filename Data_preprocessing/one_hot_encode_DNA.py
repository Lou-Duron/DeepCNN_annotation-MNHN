#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Feb 22 16:18 2022
@author: Lou Duron

This program takes DNA sequences in hdf5 format and
creates int8 numpy arrays of the sequence one_hot encoded

Example of use :
python one_hot_encode_DNA.py -d Data/DNA/HS38/hdf5
"""

import argparse
import os
import h5py
import numpy as np
import sys
sys.path.insert(0,'..')
from ModuleLibrary.utils import one_hot_DNA

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory',
                        help="Directory containing the DNA files in hdf5 format")
    return parser.parse_args()

def main():
    args = parse_arguments()

    files = os.listdir(args.directory)
    out_dir = args.directory.replace('hdf5','one_hot')

    try:
        os.mkdir(out_dir)
    except:
        print("Overwriting")

    for file in files:
        f = h5py.File(f'{args.directory}/{file}','r')
        DNA = np.array(f['data'])
        f.close()
        DNA = DNA.reshape(DNA.shape[0],)
        DNA = DNA.astype('int8')

        OH = one_hot_DNA(DNA)

        name = file.replace('.hdf5','')

        np.save(f'{out_dir}/{name}.npy', OH)

if __name__ == '__main__':
    main()