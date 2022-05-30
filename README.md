Genome wide annotation with deep CNN
===
- [name=Lou Duron]

This repository contains two tools, DeepGAP and DeepGATE, developped during my compolsory master's internship at the Muséum National d'Histoire Naturelle of Paris. In addition of this tools, a set of scripts is given in order to pre-process data downloaded from RefSeq.

DeepGAP (Deep Genome Annotation Package) aims to propose new solution for eukariotic ab initio genome wide annotation using deep convolutional neural networks. It has been sucessfully used for protein coding gene predicition in primate genomes (results in master's thesis), but has been developped to be as flexible as possible and could be used in many other ways. Please note that this tool is not finished and intends to be a basis for a complete genome wide annotation tool.

DeepGATE (Deep Genome Annotation Tool Explorer) is a tool developed to unvail the black box that is a deep CNN. By !!!!!!, it allows the user to identify sequences motifs used by the model to make its prediction. ????

## Configuration

- **Python** 3.8
- **Biopython** 1.78
- **Tensoflow** 2.5.0
- **Keras** 2.5.0
- **Pandas** 1.3.5
- **Numpy** 1.19.5
- **H5py** 3.1.0
- **Matplotlib** 3.5.0

If your computer is equiped with GPU :
- **Tensoflow-gpu** 2.5.0
- **Cudatoolkit** 11.3.1
- **Cudnn** 8.2.1.32
