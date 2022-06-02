#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jan 11 10:15 2022
@author: Lou Duron

This program take an annotation file in gff format and creates
as many files at gff format that there are chomosome with the
relevant annotations.

Example of use :
python get_annotation_from_gff.py -i Data/Raw_data/GRCh38_latest_genomic.gff -s HS38
"""

import os
import re
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help="Input file in gff format")     
    parser.add_argument('-s', '--species',
                        help="Species name")                 
    return parser.parse_args()

def main():
    args = parse_arguments()
    species = args.species
    chr_list = []
    ID_list = []
    files = []

    for element in os.listdir(f'../Data/DNA/{species}/fasta'):
        if re.match(r'chr\w+.?\.fa', element):
            num = re.search('chr\w+.?\.', element)
            chr_list.append(num.group(0)) 

    
    for chr in chr_list:
        with open(f'../Data/DNA/{species}/fasta/{chr}fa') as f:
            for line in f:
                ID = line.split(' ')[0]
                ID = ID.replace('>','')
                ID_list.append(ID)
                break

    try:
        os.mkdir(f'../Data/Annotations/{species}')
    except:
        print("\nOverwriting\n")
    try:
        os.mkdir(f'../Data/Annotations/{species}/annot')
    except:
        print("\nOverwriting\n")

    
    for chr in chr_list:
        file = open(f"../Data/Annotations/{species}/annot/{chr}gff", 'w')
        file.write("seqid\tsource\ttype\tstart\tstop\tscore\tstrand\tphase\tattibutes\n")
        files.append(file)

    with open(args.input, 'r') as f:
        for line in f:
            for j, id in enumerate(ID_list):
                if line.startswith(id):
                    files[j].write(line)
    for file in files:
        file.close()
            
if __name__ == '__main__':
    main()