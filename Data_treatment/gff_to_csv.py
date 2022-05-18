#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:07 2022
@author: lou

Creates an annotation file(.csv) from gff files

Example of use :
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
    sp = ['Maca',#
                #'HS37',#
                'Call',# 
                'LeCa',#
                'PanP',#
                'Asia',#
                'ASM2',#
                'Clin',# 
                'Kami',#
                'Mmul',#
                'Panu',#
                'Tgel'#
                ]
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
