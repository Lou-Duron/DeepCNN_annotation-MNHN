#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:23 2022
@author: lou
"""     

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization, Add, Bidirectional
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling1D, MaxPooling2D, ConvLSTM2D
from tensorflow.keras.layers import Reshape, Concatenate, ReLU, LSTM, Conv1D
from ModuleLibrary.blocks import ConvBlock, DenseDilatedConvBlock
from tensorflow.keras import Input

def Model_dic(window):
   dic = {}
   #dic['Conv_BiLSTM'] = Conv_BiLSTM(window)
   #dic['BiLSTM'] = BiLSTM(window)
   #dic['BasenjiBlocks'] = BasenjiBlocks(window)
   dic['Conv'] = Conv_DNA(window)
   dic['Conv_prot'] = Conv_prot(window)
   dic['Conv_prot2'] = Conv_prot2(window)
   dic['Conv_prot3'] = Conv_prot3(window)
   dic['Conv_prot4'] = Conv_prot4(window)
   dic['Conv_prot5'] = Conv_prot5(window)
   dic['Conv_prot6'] = Conv_prot6(window)
   dic['Conv_prot7'] = Conv_prot7(window)

   return dic


def Conv_prot(window):
    
   prot = Input(shape=(window, 21, 1))

   x = Conv2D(64, (1,21),padding='valid')(prot)
   x = ReLU()(x)

   x = Conv2D(64, (4,1),padding='same')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)

   x = Conv2D(128, (4,1),padding='same')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)

   x = Flatten()(x)

   x = Dense(128, activation = 'relu')(x)
   x = Dropout(rate = 0.1)(x)

   output = Dense(1, activation='sigmoid')(x)

   model = Model(prot, output)
    
   return model

def Conv_prot2(window):
    
   prot = Input(shape=(window, 21, 1))

   x = Conv2D(64, (1,21),padding='valid')(prot)
   x = ReLU()(x)

   x = Conv2D(64, (4,1),padding='same')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)
   x = MaxPooling2D((2,1))(x)

   x = Conv2D(128, (4,1),padding='same')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)
   x = MaxPooling2D((2,1))(x)

   x = Flatten()(x)

   x = Dense(128, activation = 'relu')(x)
   x = Dropout(rate = 0.1)(x)

   output = Dense(1, activation='sigmoid')(x)

   model = Model(prot, output)
    
   return model

def Conv_prot3(window):
    
   prot = Input(shape=(window, 21, 1))

   x = Conv2D(64, (1,21),padding='valid')(prot)
   x = ReLU()(x)

   x = Conv2D(64, (2,1),padding='same')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)
   x = MaxPooling2D((2,1))(x)

   x = Conv2D(128, (2,1),padding='same')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)
   x = MaxPooling2D((2,1))(x)

   x = Flatten()(x)

   x = Dense(128, activation = 'relu')(x)
   x = Dropout(rate = 0.1)(x)

   output = Dense(1, activation='sigmoid')(x)

   model = Model(prot, output)
    
   return model

def Conv_prot4(window):
    
   prot = Input(shape=(window, 21, 1))

   x = Conv2D(64, (3,7),padding='valid')(prot)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)

   x = Conv2D(64, (3,7),padding='valid')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)

   x = Conv2D(128, (3,7),padding='valid')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)

   x = Flatten()(x)

   x = Dense(128, activation = 'relu')(x)
   x = Dropout(rate = 0.1)(x)

   output = Dense(1, activation='sigmoid')(x)

   model = Model(prot, output)
    
   return model   

def Conv_prot5(window):
    
   prot = Input(shape=(window, 21, 1))

   x = Conv2D(64, (1,21),padding='valid')(prot)
   x = ReLU()(x)

   x = Conv2D(64, (2,1),padding='same')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)

   x = Conv2D(128, (4,1),padding='same')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)

   x = Conv2D(128, (10,1),padding='same')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)

   x = Flatten()(x)

   x = Dense(128, activation = 'relu')(x)
   x = Dropout(rate = 0.1)(x)

   output = Dense(1, activation='sigmoid')(x)

   model = Model(prot, output)
    
   return model

def Conv_prot6(window):
    
   prot = Input(shape=(window, 21, 1))

   x = Conv2D(64, (1,21),padding='valid')(prot)
   x = ReLU()(x)

   x = Conv2D(64, (10,1),padding='same')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)
   x = MaxPooling2D((2,1))(x)

   x = Conv2D(128, (5,1),padding='same')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)
   x = MaxPooling2D((2,1))(x)

   x = Flatten()(x)

   x = Dense(128, activation = 'relu')(x)
   x = Dropout(rate = 0.1)(x)

   x = Dense(64, activation = 'relu')(x)
   x = Dropout(rate = 0.1)(x)

   output = Dense(1, activation='sigmoid')(x)

   model = Model(prot, output)
    
   return model

def Conv_prot7(window):
    
   prot = Input(shape=(window, 21, 1))

   x = Conv2D(128, (1,21),padding='valid')(prot)
   x = ReLU()(x)

   x = Conv2D(128, (2,1),padding='same')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)
   x = MaxPooling2D((2,1))(x)

   x = Conv2D(256, (2,1),padding='same')(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)
   x = MaxPooling2D((2,1))(x)

   x = Flatten()(x)

   x = Dense(256, activation = 'relu')(x)
   x = Dropout(rate = 0.1)(x)

   output = Dense(1, activation='sigmoid')(x)

   model = Model(prot, output)
    
   return model


def Conv_BiLSTM_DNA(window):
   input = Input(shape=(window, 4, 1))

   x = Conv1D(16, 3, padding='valid', activation='relu')(input)
   x = Dropout(0.2)(x)
   x = MaxPooling1D((2))(x)
   x = Conv1D(32, 3, padding='valid', activation='relu')(x)
   x = Dropout(0.2)(x)
   x = MaxPooling1D((2))(x)
   x = Reshape((x.shape[1], x.shape[3]))(x)
   x = Bidirectional(LSTM(64, return_sequences=True))(x)
   x = Bidirectional(LSTM(64))(x)

   output = Dense(1, activation='sigmoid')(x)

   model = Model(input, output)
   return model

def BiLSTM_PRED(window):
   dna = Input(shape=(window, 4, 1))
   pred = Input(shape=(window, 6, 1))

   x = Concatenate(axis=2)([dna,pred])
   x = Reshape((window, 10))(x)
   x = Bidirectional(LSTM(64, return_sequences=True))(x)
   x = Bidirectional(LSTM(64))(x)

   output = Dense(1, activation='sigmoid')(x)

   model = Model([dna,pred], output)
   return model

def BasenjiBlocks_PRED(window):
   dna = Input(shape=(window, 4, 1))
   pred = Input(shape=(window, 2, 1))
 
   input = Concatenate(axis=2)([dna,pred])
  
   conv1 = ConvBlock(input, 32, 6)
   conv2 = ConvBlock(conv1, 32, 6)
   conv3 = ConvBlock(conv2, 64, 6)
 
   dil1 = DenseDilatedConvBlock(conv3, filters=64, kernel=6, dilation_rate=2)
   dil2 = DenseDilatedConvBlock(dil1, filters=64, kernel=6, dilation_rate=4)
   dil3 = DenseDilatedConvBlock(dil2, filters=64, kernel=6, dilation_rate=8)
   dil4 = DenseDilatedConvBlock(dil3, filters=64, kernel=6, dilation_rate=16)
   dil5 = DenseDilatedConvBlock(dil4, filters=64, kernel=6, dilation_rate=32)
 
   x = Flatten()(dil5)
   x = Dense(64, activation='relu')(x)
   x = Dropout(0.1)(x)
   output = Dense(1, activation='sigmoid')(x)
 
   model = Model([dna, pred], output)
 
   return model
 
def Conv_DNA(window):
    
   dna = Input(shape=(window, 4, 1))

   conv = Conv2D(16, (1,4), input_shape=(window,4,1),padding='valid')(dna)
   conv = ReLU()(conv)

   conv1 = Conv2D(32, (3,1),dilation_rate=(2,1),padding='same')(conv)
   conv1 = ReLU()(conv1)
   conv1 = Dropout(rate = 0.1)(conv1)

   conv2 = Conv2D(64, (6,1),dilation_rate=(4,1),padding='same')(conv1)
   conv2 = ReLU()(conv2)
   conv2 = Dropout(rate = 0.1)(conv2)

   conv3 = Conv2D(128, (6,1),dilation_rate=(8,1),padding='same')(conv2)
   conv3 = ReLU()(conv2)
   conv3 = Dropout(rate = 0.1)(conv2)

   x = Flatten()(conv3)

   z = Dense(256, activation = 'relu')(x)

   output = Dense(window, activation='sigmoid')(z)

   model = Model(dna, output)
    
   return model