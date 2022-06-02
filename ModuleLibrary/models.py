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
from tensorflow.keras import Input

def Model_dic(window):
   dic = {}
   dic['Conv'] = Conv_DNA(window)
   dic['Conv2'] = Conv2(window)
   dic['Conv3'] = Conv_3(window)

   return dic


def Conv_3(window):
    
   dna = Input(shape=(window, 4, 1))

   x = Conv2D(16, (1,4),padding='valid')(dna)
   x = ReLU()(x)

   x = Conv2D(32, (3,1),padding='same')(x)
   x = BatchNormalization()(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)
   x = MaxPooling2D((2,1))(x)

   x = Conv2D(32, (6,1),padding='same')(x)
   x = BatchNormalization()(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)
   x = MaxPooling2D((2,1))(x)

   x = Conv2D(64, (6,1),padding='same')(x)
   x = BatchNormalization()(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)
   x = MaxPooling2D((2,1))(x)

   x = Conv2D(64, (12,1),padding='same')(x)
   x = BatchNormalization()(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)
   x = MaxPooling2D((2,1))(x)

   x = Conv2D(128, (12,1),padding='same')(x)
   x = BatchNormalization()(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)
   x = MaxPooling2D((2,1))(x)

   x = Conv2D(128, (24,1),padding='same')(x)
   x = BatchNormalization()(x)
   x = ReLU()(x)
   x = Dropout(rate = 0.2)(x)
   x = MaxPooling2D((2,1))(x)

   x = Flatten()(x)

   x = Dense(128, activation = 'relu')(x)
   x = Dense(64, activation = 'relu')(x)
   x = Dense(64, activation = 'relu')(x)

   output = Dense(1, activation='sigmoid')(x)

   model = Model(dna, output)
    
   return model

def Conv2(window):
    
   dna = Input(shape=(window, 4, 1))

   conv = Conv2D(16, (1,4), input_shape=(window,4,1),padding='valid')(dna)
   conv = ReLU()(conv)

   conv1 = Conv2D(32, (3,1),padding='same')(conv)
   conv1 = ReLU()(conv1)
   conv1 = Dropout(rate = 0.1)(conv1)
   conv1 = MaxPooling2D((2,1))(conv1)

   conv2 = Conv2D(64, (6,1),padding='same')(conv1)
   conv2 = ReLU()(conv2)
   conv2 = Dropout(rate = 0.1)(conv2)
   conv2 = MaxPooling2D((2,1))(conv2)

   conv3 = Conv2D(128, (12,1),padding='same')(conv2)
   conv3 = ReLU()(conv3)
   conv3 = Dropout(rate = 0.1)(conv3)

   x = Flatten()(conv3)

   z = Dense(128, activation = 'relu')(x)

   output = Dense(1, activation='sigmoid')(z)

   model = Model(dna, output)
    
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

   output = Dense(1, activation='sigmoid')(z)

   model = Model(dna, output)
    
   return model