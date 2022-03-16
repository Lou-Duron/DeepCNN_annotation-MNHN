#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:07 2022
@author: lou

Creates an annotation file(.csv) from gff files

python Data_treatment/gff_to_csv.py  -o RNA -t mRNA
"""
import argparse
import pandas as pd
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',
                        help="output")
    parser.add_argument('-t', '--type',
                        help="type of the desired annotation CDS, exon or gene")
    return parser.parse_args()

def main():
    args = parse_arguments()
    sp = ['Maca', # 88 artefacts
          'HS37', # 6 artefacts
          'Call', # 11 artefacts
          'LeCa', # 1 Artefact
          'PanP', # 34 artefacts
          'Asia', # 13 artefacts + delete
          'ASM2', #  416 artefacts + delete
          'ASM7', # 32 artefacts
          'Clin', # 26 artefacts <--- Pred
          'Kami', # 49 artefacts
          'Mmul', # 4 artefacts
          'Panu', # 58 artefacts <--- Pred2
          'Tgel', # 212 artefacts
          'Cani',
          'Dani',
          'Equi',
          'Feli',
          'Gall',
          'MusM',
          'Orni']
    for species in sp:
        files = os.listdir(f'Data/Annotations/{species}/annot')
        out_file = open(f'Data/Annotations/{species}/{args.output}.csv','w')
        out_file.write("chr,start,stop,strand\n")

        for file in files:
            print(f'Converting the file : {file}')
            df = pd.read_csv(f'Data/Annotations/{species}/annot/{file}', sep = '\t', dtype={5: str})
            chr_name = file.replace('.gff3','')
            chr_name = file.replace('.gff','')
            if args.type == 'gene':
                df1 = df[df.type == 'gene']
                df2 = df[df.type == 'pseudogene']
                df = pd.concat([df1,df2])
            else :
                df = df[df.type == args.type]
            for start, stop, strand in zip(df.start, df.stop, df.strand):
                out_file.write(f"{chr_name},{start},{stop},{strand}\n")
        out_file.close()

if __name__ == '__main__':
    main()
