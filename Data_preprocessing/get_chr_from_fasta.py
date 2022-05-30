#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jan 11 09:06 2022
@author: Lou Duron

This program take an genome in fasta format and creates
as many files that there are chromosome.

Example of use :
python get_chr_from_fasta.py -i  Data/Raw_data/GRCh38_latest_genomic.fna -s HS38
"""

import os
import re
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help="input file in fasta format")
    parser.add_argument('-s', '--species',
                        help="Species name")                   
    return parser.parse_args()

def main():
    args = parse_arguments()
    out_file = None
    with open(args.input, 'r') as f:
        for line in f:
            if line.startswith(">"):
                if out_file is not None:
                    out_file.close()
                    out_file = None
                if line.startswith(">NC"):
                    match = re.findall(r'chromosome \w*,', line)
                    if len(match) > 0:
                        chr = match[0].replace('chromosome ','')
                        prefix = chr.replace(',','')
                        print(prefix)
                        path = f'Data/DNA/{args.species}'
                        try:
                            os.mkdir(path)
                        except:
                            print("Overwriting")
                        try:
                            os.mkdir(f'{path}/fasta')
                        except:
                            print("Overwriting")
                        out_file = open(f"{path}/fasta/chr{prefix}.fa", 'w')
            if out_file is not None:
                out_file.write(line)
            
if __name__ == '__main__':
    main()
