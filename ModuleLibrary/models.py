#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:23 2022
@author: lou
"""     

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization, Add
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Concatenate, ReLU
from ModuleLibrary.blocks import ConvBlock, DenseDilatedConvBlock
from tensorflow.keras import Input

def Model_dic(window):
    dic = {}
    dic['myModel1'] = myModel1(window)
    #dic['test'] = test(window)  
    #dic['testBasenji'] = testBasenji(window)
    #dic['testBasenji2'] = testBasenji2(window)
    #dic['testBasenji3'] = testBasenji3(window)
    #dic['testBasenji4'] = testBasenji4(window)
    #dic['testBasenji5'] = testBasenji5(window)

    return dic

def BasenjiDNA(window):
    input = Input(shape=(window, 4, 1))

    conv1 = Conv2D(16, (5,4), padding='valid')(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv1 = MaxPooling2D((2,1),padding='same')(conv1)

    conv2 = Conv2D(32, (5,1), padding='valid')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    conv2 = MaxPooling2D((2,1),padding='same')(conv2)

    conv3 = Conv2D(64, (5,1), padding='valid')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    conv3 = MaxPooling2D((2,1),padding='same')(conv3)

    dil1 = Conv2D(32, (3,1), dilation_rate=(2,1),padding='same')(conv3)
    dil1 = BatchNormalization()(dil1)
    dil1 = ReLU()(dil1)
    dil1 = Conv2D(64, (1,1),padding='same')(dil1)
    dil1 = BatchNormalization()(dil1)
    dil1 = Dropout(0.3)(dil1)
    dil1 = Add()([dil1, conv3])

    dil2 = Conv2D(32, (3,1), dilation_rate=(4,1),padding='same')(dil1)
    dil2 = BatchNormalization()(dil2)
    dil2 = ReLU()(dil2)
    dil2 = Conv2D(64, (1,1),padding='same')(dil2)
    dil2 = BatchNormalization()(dil2)
    dil2 = Dropout(0.3)(dil2)
    dil2 = Add()([dil1, dil2])

    dil3 = Conv2D(32, (3,1), dilation_rate=(8,1),padding='same')(dil2)
    dil3 = BatchNormalization()(dil3)
    dil3 = ReLU()(dil3)
    dil3 = Conv2D(64, (1,1),padding='same')(dil3)
    dil3 = BatchNormalization()(dil3)
    dil3 = Dropout(0.3)(dil3)
    dil3 = Add()([dil2, dil3])

    dil4 = Conv2D(32, (3,1), dilation_rate=(16,1),padding='same')(dil3)
    dil4 = BatchNormalization()(dil4)
    dil4 = ReLU()(dil4)
    dil4 = Conv2D(64, (1,1),padding='same')(dil4)
    dil4 = BatchNormalization()(dil4)
    dil4 = Dropout(0.3)(dil4)
    dil4 = Add()([dil3, dil4])

    dil5 = Conv2D(32, (3,1), dilation_rate=(32,1),padding='same')(dil4)
    dil5 = BatchNormalization()(dil5)
    dil5 = ReLU()(dil5)
    dil5 = Conv2D(64, (1,1),padding='same')(dil5)
    dil5 = BatchNormalization()(dil5)
    dil5 = Dropout(0.3)(dil5)
    dil5 = Add()([dil4, dil5])

    conv = Conv2D(128,(1,1), activation='relu')(dil5)
    conv = Dropout(0.05)(conv)

    x = Flatten()(conv)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(input, output)

    return model

def testBasenji(window):
   dna = Input(shape=(window, 4, 1))
   pred = Input(shape=(window, 4, 1))
 
   input = Concatenate(axis=2)([dna,pred])
  
   conv1 = ConvBlock(input, 32, 6)
   conv2 = ConvBlock(conv1, 32, 6)
   conv3 = ConvBlock(conv2, 64, 6)
 
   dil1 = DenseDilatedConvBlock(conv3, filters=64, kernel=6, dilation_rate=2)
   dil2 = DenseDilatedConvBlock(dil1, filters=64, kernel=6, dilation_rate=4)
   dil3 = DenseDilatedConvBlock(dil2, filters=64, kernel=6, dilation_rate=8)
   dil4 = DenseDilatedConvBlock(dil3, filters=64, kernel=6, dilation_rate=16)
   dil5 = DenseDilatedConvBlock(dil4, filters=64, kernel=6, dilation_rate=32)
 
   x = Conv2D(128, (1,1))(dil5)
   x = Dropout(0.05)(x)
   x = Flatten()(x)
   output = Dense(1, activation='sigmoid')(x)
 
   model = Model([dna, pred], output)
 
   return model
 
def testBasenji2(window):
   dna = Input(shape=(window, 4, 1))
   pred = Input(shape=(window, 4, 1))
 
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
 
def testBasenji3(window):
   dna = Input(shape=(window, 4, 1))
   pred = Input(shape=(window, 4, 1))
 
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
   x = Dense(2, activation='relu')(x)
   output = Dense(1, activation='sigmoid')(x)
 
   model = Model([dna, pred], output)
 
   return model
 
def testBasenji4(window):
   dna = Input(shape=(window, 4, 1))
   pred = Input(shape=(window, 4, 1))
 
   input = Concatenate(axis=2)([dna,pred])
  
   conv1 = ConvBlock(input, 32, 6)
   conv2 = ConvBlock(conv1, 32, 12)
   conv3 = ConvBlock(conv2, 64, 18)
 
   dil1 = DenseDilatedConvBlock(conv3, filters=64, kernel=6, dilation_rate=2)
   dil2 = DenseDilatedConvBlock(dil1, filters=64, kernel=6, dilation_rate=4)
   dil3 = DenseDilatedConvBlock(dil2, filters=64, kernel=6, dilation_rate=8)
   dil4 = DenseDilatedConvBlock(dil3, filters=64, kernel=6, dilation_rate=16)
   dil5 = DenseDilatedConvBlock(dil4, filters=64, kernel=6, dilation_rate=32)
 
   x = Conv2D(128, (1,1))(dil5)
   x = Dropout(0.05)(x)
   x = Flatten()(x)
   output = Dense(1, activation='sigmoid')(x)
 
   model = Model([dna, pred], output)
 
   return model
 
def testBasenji5(window):
   dna = Input(shape=(window, 4, 1))
   pred = Input(shape=(window, 4, 1))
 
   input = Concatenate(axis=2)([dna,pred])
  
   conv1 = ConvBlock(input, 32, 6)
   conv2 = ConvBlock(conv1, 32, 6)
   conv3 = ConvBlock(conv2, 64, 6)
   conv4 = ConvBlock(conv3, 64, 6)
   conv5 = ConvBlock(conv4, 128, 6)
 
   dil1 = DenseDilatedConvBlock(conv5, filters=128, kernel=6, dilation_rate=2)
   dil2 = DenseDilatedConvBlock(dil1, filters=128, kernel=6, dilation_rate=4)
   dil3 = DenseDilatedConvBlock(dil2, filters=128, kernel=6, dilation_rate=8)
   dil4 = DenseDilatedConvBlock(dil3, filters=128, kernel=6, dilation_rate=16)
   dil5 = DenseDilatedConvBlock(dil4, filters=128, kernel=6, dilation_rate=32)
   dil6 = DenseDilatedConvBlock(dil5, filters=128, kernel=6, dilation_rate=64)
   dil7 = DenseDilatedConvBlock(dil6, filters=128, kernel=6, dilation_rate=128)
 
   x = Conv2D(256, (1,1))(dil7)
   x = Dropout(0.05)(x)
   x = Flatten()(x)
   output = Dense(1, activation='sigmoid')(x)
 
   model = Model([dna, pred], output)
 
   return model



def test(window):
    dna = Input(shape=(window, 4, 1))
    pred = Input(shape=(window, 4, 1))
    
    conv1 = Conv2D(32, kernel_size=(6,4), name= 'DNAconv1',activation='relu', padding='valid')(dna)
    conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 =  MaxPooling2D((2,1),padding='same')(conv1)

    conv2 = Conv2D(32, kernel_size=(6,1), name= 'DNAconv2', activation='relu', padding='valid')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 =  MaxPooling2D((2,1),padding='same')(conv2)

    conv3 = Conv2D(32, kernel_size=(6,1), name= 'DNAconv3', activation='relu', padding='valid')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3_dna =  MaxPooling2D((2,1),padding='same')(conv3)

    conv1 = Conv2D(32, kernel_size=(6,1), name= 'PREDconv1', activation='relu', padding='valid')(pred)
    conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 =  MaxPooling2D((2,1),padding='same')(conv1)

    conv2 = Conv2D(32, kernel_size=(6,1), name= 'PREDconv2', activation='relu', padding='valid')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 =  MaxPooling2D((2,1),padding='same')(conv2)

    conv3 = Conv2D(32, kernel_size=(6,1), name= 'PREDconv3', activation='relu', padding='valid')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3_pred =  MaxPooling2D((2,1),padding='same')(conv3)

    x = Concatenate(axis=2)([conv3_dna,conv3_pred])

    dil1 = Conv2D(32, kernel_size=(6,5), name= 'PREDdil1', activation='relu', padding='same', dilation_rate=(2, 1))(x)
    dil1 = Dropout(0.2)(dil1)
    dil1 = BatchNormalization()(dil1)

    dil2 = Conv2D(32, kernel_size=(6,5), name= 'PREDdil2', activation='relu', padding='same', dilation_rate=(4, 1))(x)
    dil2 = Dropout(0.2)(dil2)
    dil2 = BatchNormalization()(dil2)

    dil3 = Conv2D(32, kernel_size=(6,5), name= 'PREDdil3', activation='relu', padding='same', dilation_rate=(8, 1))(x)
    dil3 = Dropout(0.2)(dil3)
    dil3 = BatchNormalization()(dil3)

    dil4 = Conv2D(32, kernel_size=(6,5), name= 'PREDdil4', activation='relu', padding='same', dilation_rate=(16, 1))(x)
    dil4 = Dropout(0.2)(dil4)
    dil4 = BatchNormalization()(dil4)

    dil5 = Conv2D(32, kernel_size=(6,5), name= 'PREDdil5', activation='relu', padding='same', dilation_rate=(32, 1))(x)
    dil5 = Dropout(0.2)(dil5)
    dil5 = BatchNormalization()(dil5)

    x = Add()([dil1, dil2, dil3, dil4, dil5])

    x = Flatten()(x)

    z = Dense(64, activation = 'relu')(x)

    output = Dense(1, activation='sigmoid')(z)

    model = Model([dna, pred], output)

    return model

def myModel1(window):
    
    model = Sequential()

    model.add(Conv2D(32, (6,4), activation='relu',input_shape=(window,4,1),padding='valid'))
    model.add(MaxPooling2D((2, 1),padding='same'))
    model.add(Dropout(rate = 0.2))

    model.add(Conv2D(64, (6, 1), activation='relu',padding='same'))
    model.add(MaxPooling2D((2, 1),padding='same'))
    model.add(Dropout(rate=0.2))

    model.add(Conv2D(128, (6, 1), activation='relu',padding='same'))
    model.add(MaxPooling2D((2, 1),padding='same'))
    model.add(Dropout(rate=0.2))
    
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.2))

    model.add(Dense(1, activation='sigmoid'))
    
    return model
