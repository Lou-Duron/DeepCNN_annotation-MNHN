#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jan 12 14:07 2022
@author: Lou Duron

This program creates an annotation file(.csv) from gff files

Example of use :
python gff_to_csv.py -o RNA -t mRNA -s HS38
"""

import argparse
import pandas as pd
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',
                        help="output file name")
    parser.add_argument('-t', '--type',
                        help="type of the desired annotation CDS, exon, mRNA or gene")
    parser.add_argument('-s', '--species',
                        help="Species name")                   
    return parser.parse_args()

def main():
    args = parse_arguments()
    files = os.listdir(f'../Data/Annotations/{args.species}/annot')
    out_file = open(f'../Data/Annotations/{args.species}/{args.output}.csv','w')
    out_file.write("chr,start,stop,strand\n")
    for file in files:
        print(f'Converting the file : {file}')
        df = pd.read_csv(f'../Data/Annotations/{args.species}/annot/{file}',
                         sep = '\t', dtype={5: str})
        chr_name = file.replace('.gff3','')
        chr_name = file.replace('.gff','')
        df = df[df.type == args.type]
        for start, stop, strand in zip(df.start, df.stop, df.strand):
            out_file.write(f"{chr_name},{start},{stop},{strand}\n")
    out_file.close()

if __name__ == '__main__':
    main()
