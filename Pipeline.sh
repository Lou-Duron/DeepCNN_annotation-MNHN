#!/bin/bash


python Prediction_one_chr.py -p multi3_EXON_start_w50 -s Panu -c 1 -v -r multi3_exon_start_w50_reverse

python Prediction_one_chr.py -p multi3_EXON_stop_w50 -s Panu -c 1 -r multi3_exon_stop_w50
python Prediction_one_chr.py -p multi3_EXON_stop_w50 -s Panu -c 1 -v -r multi3_exon_stop_w50_reverse
