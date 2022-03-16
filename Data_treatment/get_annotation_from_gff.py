#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:15 2022
@author: lou

This program take an annotation file in gff format and creates
as many files at gff format that there are chomosome with the
relevant annotations.

python Data_treatment/get_annotation_from_gff.py -i Data/Raw_data/GCF_009663435.1_Callithrix_jacchus_cj1700_1.1_genomic.gff -f Data/DNA/Callithrix/fasta -o Data/Annotations/Callithrix/annot
python Data_treatment/get_annotation_from_gff.py -i Data/Raw_data/GCF_013052645.1_Mhudiblu_PPA_v0_genomic.gff -f Data/DNA/Callithrix/fasta -o Data/Annotations/Callithrix/annot
"""

import os
import re


def main():
    files = ['Cani','Dani','Equi','Feli','Gall','MusM','Orni']
    for species in files:
        chr_list = []
        for element in os.listdir(f'Data/DNA/{species}/fasta'):
            if re.match(r'chr\w+.?\.fa', element):
                num = re.search('chr\w+.?\.', element)
                chr_list.append(num.group(0)) 

        ID_list = []
        for chr in chr_list:
            with open(f'Data/DNA/{species}/fasta/{chr}fa') as f:
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
        with open(f'Data/Raw_data/animals/{species}.gff', 'r') as f:
            for line in f:
                for j, id in enumerate(ID_list):
                    if line.startswith(id):
                        files[j].write(line)
        for file in files:
            file.close()
            
if __name__ == '__main__':
    main()