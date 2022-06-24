#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Apr 27 10:52 2022
@author: Lou Duron

This module contains DeepGATE, a tool made to explore
DeepGAP's neural networks to indentify sequences motifs
used by the CNN for the preditction.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
from tensorflow.keras.models import Model, load_model

import sys
sys.path.insert(0,'..')

from ModuleLibrary.utils import sliding_window_view, padding_slidding

class Explorer():

    def __init__(self, model_path, custom_objects):
        self.model = load_model(model_path, custom_objects=custom_objects)
        self.model_w = h5py.File(model_path,'r')
        self.layers = []
        self.build_model()
        
    def build_model(self):
        for layer in self.model.get_config()['layers']:

            if layer['class_name'].startswith('Conv'):
                name = layer['config']['name']
                kernel = layer['config']['kernel_size']
                padding = layer['config']['padding']
                self.layers.append(Conv(name, self.model_w, kernel,
                                        padding))

            elif layer['class_name'].startswith('MaxPooling'):
                name = layer['config']['name']
                pool_size = layer['config']['pool_size']
                self.layers.append(MaxPooling(name, pool_size))

            elif layer['class_name'].startswith('Dense'):
                name = layer['config']['name']
                units = layer['config']['units']
                self.layers.append(Dense(name, self.model_w, units))

            elif layer['class_name'].startswith('Flatten'):
                name = layer['config']['name']
                self.layers.append(Flatten(name))

            elif layer['class_name'].startswith('Input'):
                name = layer['config']['name']
                self.layers.append(Input(name))
            else:
                continue

    def get_inputs_and_outputs(self, data):
        for i, layer in enumerate(self.layers):
            layer.set_output(self.model, data)
            if i > 0:
                layer.input = self.layers[i-1].output
            else:
                layer.input = np.zeros(1)

    def explore(self, data):
        data = data.reshape(1,data.shape[0], data.shape[1],1)
        self.get_inputs_and_outputs(data)
        layers_rev = list(reversed(self.layers))
        for i, layer in enumerate(layers_rev):
            layer.compute_contrib()
            if i != len(self.layers) - 1:
                layers_rev[i+1].output_contrib = layer.input_contrib

        self.plot()
       

    def plot(self):

        cim = plt.imread("DeepGATE/colorbar.png")
        cim = cim[cim.shape[0]//2, 50:390, :]
        cmap = mcolors.ListedColormap(cim)
        plt.figure(figsize=(34,1), dpi= 200)
        plt.imshow(self.layers[0].input_contrib, cmap=cmap, aspect='auto',
                   vmin=-1, vmax=1)
        plt.xticks(np.arange(0, 2001,10))
        plt.yticks([0,1,2,3], ['a','t','g','c'])
        plt.colorbar()
        plt.show()


class Layer():

    def __init__(self, name):
        self.name = name

    def compute_contrib(self):
        pass

    def set_output(self, model, data):
        self.output = np.squeeze(Model(
            inputs=model.inputs, 
            outputs=model.get_layer(name=self.name).output).predict(data))

    def set_weights(self, model_w):
        self.weights = np.squeeze(np.array(model_w['model_weights']
                                 [self.name][self.name]['kernel:0']))

    def set_bias(self, model_w):
        self.bias = np.squeeze(np.array(model_w['model_weights']
                                [self.name][self.name]['bias:0']))

class Conv(Layer):

    def __init__(self, name, model_w, kernel, padding):
        super().__init__(name)
        self.set_weights(model_w)
        self.set_bias(model_w)
        self.kernel = kernel
        self.padding = padding

    def compute_contrib(self):
        super().compute_contrib()
        self.output_contrib[self.output <= 0] *= 1
        if self.padding == 'valid':
            input = sliding_window_view(self.input, self.kernel[0], axis=0)
            input = input.flatten(order='K').reshape((input.shape[0],
                                                      input.shape[2],
                                                      input.shape[1]))
            input_contrib_tmp = np.einsum('ijk,jkl,il->ijk', 
                                        input, 
                                        self.weights, 
                                        self.output_contrib)
            self.input_contrib = np.zeros((self.input.shape[0], self.input.shape[1]))
            for i in range(input_contrib_tmp.shape[0]):
                step = input_contrib_tmp.shape[1]
                self.input_contrib[i:i+step] += input_contrib_tmp[i]
            
            
        else:
            input = padding_slidding(self.input, self.kernel[0])
            self.input_contrib = np.einsum('ijk,jkl,il->ik', 
                                           input, 
                                           self.weights, 
                                           self.output_contrib)
            

class Dense(Layer):

    def __init__(self, name, model_w, units):
        super().__init__(name)
        self.set_weights(model_w)
        self.set_bias(model_w)
        self.units = units

    def compute_contrib(self): 
        super().compute_contrib()
        if self.units > 1:
            self.output_contrib[self.output <= 0] *= 1
            self.input_contrib = np.einsum('i,ij,j->i', 
                                           self.input, 
                                           self.weights, 
                                           self.output_contrib)
        else:
            self.input_contrib = self.input * self.weights
            

class MaxPooling(Layer):

    def __init__(self, name, pool_size):
        super().__init__(name)
        self.pool_size = pool_size

    def compute_contrib(self):
        super().compute_contrib()
        self.input_contrib = np.zeros((self.input.shape[0], 
                                       self.input.shape[1])) 
        if self.input_contrib.shape[0] % 2 == 0:
            size = self.input_contrib.shape[0]
        else:
            size = self.input_contrib.shape[0] - 1
        for i in range(0,size,2):
            for j in range(self.input_contrib.shape[1]):
                if self.input[i][j] > self.input[i+1][j]:
                    self.input_contrib[i][j] = self.output_contrib[int(i/2)][j]
                else:
                    self.input_contrib[i+1][j] = self.output_contrib[int(i/2)][j]

class Flatten(Layer):

    def __init__(self, name):
        super().__init__(name)

    def compute_contrib(self):
        super().compute_contrib()
        self.input_contrib = self.output_contrib.reshape((self.input.shape[0],
                                                          self.input.shape[1]))

class Input(Layer):

    def __init__(self, name):
        super().__init__(name)

    def set_output(self, model, data):
        self.output = np.squeeze(data)

    def compute_contrib(self): 
        super().compute_contrib()
        self.input_contrib = self.output_contrib
        maxvalue = max(float(np.max(self.input_contrib)),
                       float(-np.min(self.input_contrib)))
        for i in range(len(self.input_contrib)):
            for j in range(4):
                self.input_contrib[i][j] = self.input_contrib[i][j] / maxvalue

        self.input_contrib = np.transpose(self.input_contrib)


