#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jan 7 9:22 2022
@author: Lou Duron

This program takes .fa or .fa.gz files and convert them to .hdf5 files.
If the output directory is not specified, the hdf5 files will be created in
the same directory as the fasta files.

Example of use :
python fasta_to_hdf5.py -i input_directory/ -o output_directory/
"""

import gzip
import re
import os
import argparse
import numpy as np
import h5py
from Bio import SeqIO

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help="Input derectory")
    parser.add_argument('-o', '--output',
                        help="Output directory")
    return parser.parse_args()

def converter():
    """
    Dictionnary used to convert DNA sequence into number.
    """
    dconv = {}
    dconv["N"] = 0
    dconv["n"] = 0
    dconv["A"] = 1
    dconv["a"] = 1
    dconv["T"] = 2
    dconv["t"] = 2
    dconv["G"] = 3
    dconv["g"] = 3
    dconv["C"] = 4
    dconv["c"] = 4
    dconv["K"] = 0
    dconv["f"] = 0
    dconv["M"] = 0
    dconv["R"] = 0
    dconv["Y"] = 0
    dconv["S"] = 0
    dconv["W"] = 0
    dconv["B"] = 0
    dconv["V"] = 0
    dconv["H"] = 0
    dconv["D"] = 0
    dconv["X"] = 0
    return dconv

def convert_seq(seq):
    """
    Convert a sequence.
    """
    i = 0
    L = len(seq)
    conv = converter()
    seqL = np.zeros((L, 1))
    while i < L:
        if i % 10000 == 0:
            print(f"In progress : {round(i/L*100,1)}%", end = "\r")
        seqL[i] = conv[seq[i]]
        i += 1
    return seqL, L

def fa_converter(in_file, path_to_out_file):
    '''
    Takes a .fa file converts it into an .hdf5 file.
    filenamin: the .fa file that need to be converted (or a .fa.gz file),
    one chromosome only.
    '''
    if re.match(r'.*\.fa$', os.path.basename(in_file)):
        fasta = open(in_file, 'rt')
    elif re.match(r'.*\.fa\.gz$', os.path.basename(in_file)):
        fasta = gzip.open(in_file, 'rt')
    else:
        raise ValueError("file must be a fasta file (or .fa.gz)")
        
    print(f'Converting the file : {in_file}')
    
    for seq_record in SeqIO.parse(fasta, 'fasta'):
        vout = convert_seq(seq_record.seq)[0]
    hdf5 = h5py.File(path_to_out_file)
    hdf5['data'] = vout
    hdf5.close()
    fasta.close()

def main():
    """
    Converts the .fa sequence file in a directory to .hdf5 file
    (all the file chr*.fa in the directory).
    """

    args = parse_arguments()

    in_dir = args.input
    if args.output is not None:
        out_dir = args.output
        try:
            os.mkdir(out_dir)
        except:
            print("Output directory already exists, overwriting")
    else:
        out_dir = args.input

    for element in os.listdir(in_dir):
        if re.match(r'chr\w+.?\.fa', element):
            num = re.search('chr\w+.?\.', element)
            fa_converter(os.path.join(in_dir, element),
                        os.path.join(out_dir, num.group(0) + 'hdf5'))
            
if __name__ == '__main__':
    main()
