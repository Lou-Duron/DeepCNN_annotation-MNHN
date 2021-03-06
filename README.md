Genome wide annotation with deep CNN
===
By : Lou Duron

This repository contains two tools, DeepGAP and DeepGATE, developped during my compolsory master's internship at the National Museum of Natural History of Paris. In addition of this tools, a set of scripts is given in order to pre-process data downloaded from RefSeq.

DeepGAP (Deep Genome Annotation Package) aims to propose new solutions for eukaryotic genome wide annotation using deep convolutional neural networks. It has been sucessfully used for protein coding gene and Pre-mRNA splicing site predicition in primate genomes (results in master's thesis). 

DeepGATE (Deep Genome Annotation Tool Explorer) is a tool developed to explore DeepGAP's deep CNN, allowing users to identify sequence motifs used by the model for prediciton. 

## Minimal configuration

- **Python** 3.8 
- **Biopython** 1.78
- **Tensorflow** 2.5.0
- **Keras** 2.5.0
- **Pandas** 1.3.5
- **Numpy** 1.19.5
- **H5py** 3.1.0
- **Matplotlib** 3.5.0

If your computer is equipped with GPU :
- **Tensorflow-gpu** 2.5.0
- **Cudatoolkit** 11.3.1
- **Cudnn** 8.2.1.32
