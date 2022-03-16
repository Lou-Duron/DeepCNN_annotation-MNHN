#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:06 2022
@author: lou

This program take an genome in fasta format and creates
as many files that there are chromosome.

Example of use :
python Data_treatment/get_chr_from_fasta.py
"""

import os
import re


def main():

    files = ['Amel','Cani','Dani','Equi','Feli','Gall','MusM','Orni']

    for file in files:
        out_file = None
        with open(f'Data/Raw_data/animals/{file}.fna', 'r') as f:
            print(file)
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
                            path = f'Data/DNA/{file}'
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
