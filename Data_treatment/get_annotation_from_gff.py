#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:15 2022
@author: lou

This program take an annotation file in gff format and creates
as many files at gff format that there are chomosome with the
relevant annotations.

Example of use :
python Data_treatment/get_annotation_from_gff.py
"""

import os
import re


def main():
    files = ['HS38']
    for species in files:
        chr_list = []
        for element in os.listdir(f'Data/DNA/HS37/fasta'):
            if re.match(r'chr\w+.?\.fa', element):
                num = re.search('chr\w+.?\.', element)
                chr_list.append(num.group(0)) 

        ID_list = []
        for chr in chr_list:
            with open(f'Data/DNA/HS37/fasta/{chr}fa') as f:
                for line in f:
                    ID = line.split(' ')[0]
                    ID = ID.replace('>','')
                    ID_list.append(ID)
                    break
        os.mkdir(f'Data/Annotations/{species}')
        os.mkdir(f'Data/Annotations/{species}/annot')

        files = []
        for chr in chr_list:
            file = open(f"Data/Annotations/{species}/annot/{chr}gff", 'w')
            file.write("seqid\tsource\ttype\tstart\tstop\tscore\tstrand\tphase\tattibutes\n")
            files.append(file)
        with open(f'Data/Raw_data/GRCh38_latest_genomic.gff', 'r') as f:
            for line in f:
                for j, id in enumerate(ID_list):
                    if line.startswith(id):
                        files[j].write(line)
        for file in files:
            file.close()
            
if __name__ == '__main__':
    main()