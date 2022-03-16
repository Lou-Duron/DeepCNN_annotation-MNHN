#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 15:23 2022
@author: lou
"""     

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization, Add
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, LSTM, Reshape
from tensorflow.keras import Input

def Model_dic(window):
    dic = {}
    dic['myModel1'] = myModel1(window)
    dic['myModel2'] = myModel2(window) 
    dic['dilated'] = dilated(window)
    dic['dilated2'] = dilated2(window)
    dic['dilatedValid'] = dilatedValid(window)
    dic['dilatedValid2'] = dilatedValid2(window)
    dic['dilatedValid3'] = dilatedValid3(window)
    dic['dilatedValid4'] = dilatedValid4(window)
    dic['dilated3'] = dilated3(window)
    dic['dilated4'] = dilated4(window)
    dic['dilated5'] = dilated5(window)
    dic['custom'] = custom(window)
    dic['smallcustom'] = smallcustom(window)

    return dic


def myModel1(window):
    
    model = Sequential()

    model.add(Conv2D(32, (4,4), activation='relu',input_shape=(window,4,1),padding='same'))
    model.add(MaxPooling2D((2, 1),padding='same'))
    model.add(Dropout(rate = 0.2))

    model.add(Conv2D(64, (4, 4), activation='relu',padding='same'))
    model.add(MaxPooling2D((2, 1),padding='same'))
    model.add(Dropout(rate=0.2))

    model.add(Conv2D(128, (4, 4), activation='relu',padding='same'))
    model.add(MaxPooling2D((2, 1),padding='same'))
    model.add(Dropout(rate=0.2))
    
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.2))

    model.add(Dense(1, activation='sigmoid'))
    
    return model

def myModel2(window):
    
    model = Sequential()

    model.add(Conv2D(32, (4,2), activation='relu',input_shape=(window,4,1),padding='same'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(rate = 0.2))

    model.add(Conv2D(64, (4, 2), activation='relu',padding='same'))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Dropout(rate=0.2))

    model.add(Conv2D(128, (4, 1), activation='relu',padding='same'))
    model.add(MaxPooling2D((2, 1),padding='same'))
    model.add(Dropout(rate=0.2))
    
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(rate=0.2))

    model.add(Dense(1, activation='sigmoid'))
    
    return model

def dilated(window):
    dna_input = Input(shape=(window, 4, 1))
    
    conv1 = Conv2D(32, kernel_size=(6,4), activation='relu', padding='same')(dna_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 =  MaxPooling2D((2,1),padding='same')(conv1)

    conv2 = Conv2D(64, kernel_size=(6,4), activation='relu', padding='same')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 =  MaxPooling2D((2,2),padding='same')(conv2)

    conv3 = Conv2D(128, kernel_size=(6,2), activation='relu', padding='same')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 =  MaxPooling2D((2,2),padding='same')(conv3)

    dil1 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same')(conv3)
    dil1 = Dropout(0.2)(dil1)
    dil1 = BatchNormalization()(dil1)

    dil2 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(2, 1))(dil1)
    dil2 = Dropout(0.2)(dil2)
    dil2 = BatchNormalization()(dil2)

    x = Add()([dil1, dil2])

    dil3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(4, 1))(x)
    dil3 = Dropout(0.2)(dil3)
    dil3 = BatchNormalization()(dil3)

    x = Add()([dil1, dil2, dil3])

    dil4 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(8, 1))(x)
    dil4 = Dropout(0.2)(dil4)
    dil4 = BatchNormalization()(dil4)

    x = Add()([dil1, dil2, dil3, dil4])

    dil5 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(16, 1))(x)
    dil5 = Dropout(0.2)(dil5)
    dil5 = BatchNormalization()(dil5)

    x = Add()([dil1, dil2, dil3, dil4, dil5])

    dil6 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(32, 1))(x)
    dil6 = Dropout(0.2)(dil6)
    dil6 = BatchNormalization()(dil6)

    x = Add()([dil1, dil2, dil3, dil4, dil5, dil6])

    dil7 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(64, 1))(x)
    dil7 = Dropout(0.2)(dil7)
    dil7 = BatchNormalization()(dil7)

    x = Add()([dil1, dil2, dil3, dil4, dil5, dil6, dil7])
    x = Flatten()(x)
    x = Dense(128, activation = 'relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model([dna_input], output)

    return model

def dilated2(window):
    dna_input = Input(shape=(window, 4, 1))
    
    conv1 = Conv2D(32, kernel_size=(6,4), activation='relu', padding='same')(dna_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 =  MaxPooling2D((2,1),padding='same')(conv1)

    conv2 = Conv2D(32, kernel_size=(6,4), activation='relu', padding='same')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 =  MaxPooling2D((2,2),padding='same')(conv2)

    conv3 = Conv2D(32, kernel_size=(6,2), activation='relu', padding='same')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 =  MaxPooling2D((2,2),padding='same')(conv3)

    dil1 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(2, 1))(conv3)
    dil1 = Dropout(0.2)(dil1)
    dil1 = BatchNormalization()(dil1)

    dil2 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(4, 1))(dil1)
    dil2 = Dropout(0.2)(dil2)
    dil2 = BatchNormalization()(dil2)

    x = Add()([dil1, dil2])

    dil3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(8, 1))(x)
    dil3 = Dropout(0.2)(dil3)
    dil3 = BatchNormalization()(dil3)

    x = Add()([dil1, dil2, dil3])

    dil4 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(16, 1))(x)
    dil4 = Dropout(0.2)(dil4)
    dil4 = BatchNormalization()(dil4)

    x = Add()([dil1, dil2, dil3, dil4])

    dil5 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(32, 1))(x)
    dil5 = Dropout(0.2)(dil5)
    dil5 = BatchNormalization()(dil5)

    x = Flatten()(dil5)
    x = Dense(64, activation = 'relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model([dna_input], output)

    return model

def dilatedValid(window):
    dna_input = Input(shape=(window, 4, 1))
    
    conv1 = Conv2D(32, kernel_size=(6,4), activation='relu', padding='valid')(dna_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 =  MaxPooling2D((2,1),padding='same')(conv1)

    conv2 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='valid')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 =  MaxPooling2D((2,1),padding='same')(conv2)

    conv3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='valid')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 =  MaxPooling2D((2,1),padding='same')(conv3)

    dil1 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(2, 1))(conv3)
    dil1 = Dropout(0.2)(dil1)
    dil1 = BatchNormalization()(dil1)

    dil2 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(4, 1))(dil1)
    dil2 = Dropout(0.2)(dil2)
    dil2 = BatchNormalization()(dil2)

    x = Add()([dil1, dil2])

    dil3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(8, 1))(x)
    dil3 = Dropout(0.2)(dil3)
    dil3 = BatchNormalization()(dil3)

    x = Add()([dil1, dil2, dil3])

    dil4 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(16, 1))(x)
    dil4 = Dropout(0.2)(dil4)
    dil4 = BatchNormalization()(dil4)

    x = Add()([dil1, dil2, dil3, dil4])

    dil5 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(32, 1))(x)
    dil5 = Dropout(0.2)(dil5)
    dil5 = BatchNormalization()(dil5)

    x = Flatten()(dil5)
    x = Dense(64, activation = 'relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model([dna_input], output)

    return model


def dilatedValid2(window):
    dna_input = Input(shape=(window, 4, 1))
    
    conv1 = Conv2D(8, kernel_size=(1,4), activation='relu', padding='valid')(dna_input)

    conv1 = Conv2D(16, kernel_size=(3,1), activation='relu', padding='valid')(conv1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 =  MaxPooling2D((2,1),padding='same')(conv1)

    conv2 = Conv2D(32, kernel_size=(3,1), activation='relu', padding='valid')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 =  MaxPooling2D((2,1),padding='same')(conv2)

    conv3 = Conv2D(32, kernel_size=(3,1), activation='relu', padding='valid')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 =  MaxPooling2D((2,1),padding='same')(conv3)

    conv4 = Conv2D(32, kernel_size=(3,1), activation='relu', padding='valid')(conv3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 =  MaxPooling2D((2,1),padding='same')(conv4)

    dil1 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(2, 1))(conv4)
    dil1 = Dropout(0.2)(dil1)
    dil1 = BatchNormalization()(dil1)

    dil2 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(4, 1))(dil1)
    dil2 = Dropout(0.2)(dil2)
    dil2 = BatchNormalization()(dil2)

    x = Add()([dil1, dil2])

    dil3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(8, 1))(x)
    dil3 = Dropout(0.2)(dil3)
    dil3 = BatchNormalization()(dil3)

    x = Add()([dil1, dil2, dil3])

    dil4 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(16, 1))(x)
    dil4 = Dropout(0.2)(dil4)
    dil4 = BatchNormalization()(dil4)

    x = Add()([dil1, dil2, dil3, dil4])

    dil5 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(32, 1))(x)
    dil5 = Dropout(0.2)(dil5)
    dil5 = BatchNormalization()(dil5)

    x = Flatten()(dil5)
    x = Dense(64, activation = 'relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model([dna_input], output)

    return model

def dilatedValid3(window):
    dna_input = Input(shape=(window, 4, 1))

    conv1 = Conv2D(16, kernel_size=(3,4), activation='relu', padding='valid')(dna_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 =  MaxPooling2D((2,1),padding='same')(conv1)

    conv2 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='valid')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 =  MaxPooling2D((2,1),padding='same')(conv2)

    conv3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='valid')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 =  MaxPooling2D((2,1),padding='same')(conv3)

    conv4 = Conv2D(64, kernel_size=(12,1), activation='relu', padding='valid')(conv3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 =  MaxPooling2D((2,1),padding='same')(conv4)

    dil1 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(2, 1))(conv4)
    dil1 = Dropout(0.2)(dil1)
    dil1 = BatchNormalization()(dil1)

    dil2 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(4, 1))(dil1)
    dil2 = Dropout(0.2)(dil2)
    dil2 = BatchNormalization()(dil2)

    x = Add()([dil1, dil2])

    dil3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(8, 1))(x)
    dil3 = Dropout(0.2)(dil3)
    dil3 = BatchNormalization()(dil3)

    x = Add()([dil1, dil2, dil3])

    dil4 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(16, 1))(x)
    dil4 = Dropout(0.2)(dil4)
    dil4 = BatchNormalization()(dil4)

    x = Add()([dil1, dil2, dil3, dil4])

    dil5 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(32, 1))(x)
    dil5 = Dropout(0.2)(dil5)
    dil5 = BatchNormalization()(dil5)

    x = Flatten()(dil5)
    x = Dense(64, activation = 'relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model([dna_input], output)

    return model

def dilated3(window):
    dna_input = Input(shape=(window, 4, 1))
    
    conv1 = Conv2D(32, kernel_size=(6,4), activation='relu', padding='same')(dna_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 =  MaxPooling2D((2,1),padding='same')(conv1)

    conv2 = Conv2D(32, kernel_size=(12,4), activation='relu', padding='same')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 =  MaxPooling2D((2,2),padding='same')(conv2)

    conv3 = Conv2D(32, kernel_size=(24,2), activation='relu', padding='same')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 =  MaxPooling2D((2,2),padding='same')(conv3)

    dil1 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(2, 1))(conv3)
    dil1 = Dropout(0.2)(dil1)
    dil1 = BatchNormalization()(dil1)

    dil2 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(4, 1))(dil1)
    dil2 = Dropout(0.2)(dil2)
    dil2 = BatchNormalization()(dil2)

    x = Add()([dil1, dil2])

    dil3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(8, 1))(x)
    dil3 = Dropout(0.2)(dil3)
    dil3 = BatchNormalization()(dil3)

    x = Add()([dil1, dil2, dil3])

    dil4 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(16, 1))(x)
    dil4 = Dropout(0.2)(dil4)
    dil4 = BatchNormalization()(dil4)

    x = Add()([dil1, dil2, dil3, dil4])

    dil5 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(32, 1))(x)
    dil5 = Dropout(0.2)(dil5)
    dil5 = BatchNormalization()(dil5)

    x = Flatten()(dil5)
    x = Dense(64, activation = 'relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model([dna_input], output)

    return model

def dilated4(window):
    dna_input = Input(shape=(window, 4, 1))
    
    conv1 = Conv2D(32, kernel_size=(6,4), activation='relu', padding='same')(dna_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 =  MaxPooling2D((2,1),padding='same')(conv1)

    conv2 = Conv2D(32, kernel_size=(12,4), activation='relu', padding='same')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 =  MaxPooling2D((4,2),padding='same')(conv2)

    conv3 = Conv2D(32, kernel_size=(24,2), activation='relu', padding='same')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 =  MaxPooling2D((4,2),padding='same')(conv3)

    dil1 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(2, 1))(conv3)
    dil1 = Dropout(0.2)(dil1)
    dil1 = BatchNormalization()(dil1)

    dil2 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(4, 1))(dil1)
    dil2 = Dropout(0.2)(dil2)
    dil2 = BatchNormalization()(dil2)

    x = Add()([dil1, dil2])

    dil3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(8, 1))(x)
    dil3 = Dropout(0.2)(dil3)
    dil3 = BatchNormalization()(dil3)

    x = Add()([dil1, dil2, dil3])

    dil4 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(16, 1))(x)
    dil4 = Dropout(0.2)(dil4)
    dil4 = BatchNormalization()(dil4)

    x = Add()([dil1, dil2, dil3, dil4])

    dil5 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(32, 1))(x)
    dil5 = Dropout(0.2)(dil5)
    dil5 = BatchNormalization()(dil5)

    x = Flatten()(dil5)
    x = Dense(64, activation = 'relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model([dna_input], output)

    return model

def dilated5(window): #bigger window
    dna_input = Input(shape=(window, 4, 1))
    
    conv1 = Conv2D(32, kernel_size=(6,4), activation='relu', padding='same')(dna_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 =  MaxPooling2D((2,2),padding='same')(conv1)

    conv2 = Conv2D(32, kernel_size=(12,2), activation='relu', padding='same')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 =  MaxPooling2D((4,1),padding='same')(conv2)

    conv3 = Conv2D(32, kernel_size=(12,2), activation='relu', padding='same')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 =  MaxPooling2D((2,2),padding='same')(conv3)

    conv4 = Conv2D(32, kernel_size=(24,4), activation='relu', padding='same')(conv3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 =  MaxPooling2D((4,1),padding='same')(conv4)

    dil1 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(2, 1))(conv4)
    dil1 = Dropout(0.2)(dil1)
    dil1 = BatchNormalization()(dil1)

    dil2 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(4, 1))(dil1)
    dil2 = Dropout(0.2)(dil2)
    dil2 = BatchNormalization()(dil2)

    x = Add()([dil1, dil2])

    dil3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(8, 1))(x)
    dil3 = Dropout(0.2)(dil3)
    dil3 = BatchNormalization()(dil3)

    x = Add()([dil1, dil2, dil3])

    dil4 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(16, 1))(x)
    dil4 = Dropout(0.2)(dil4)
    dil4 = BatchNormalization()(dil4)

    x = Add()([dil1, dil2, dil3, dil4])

    dil5 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(32, 1))(x)
    dil5 = Dropout(0.2)(dil5)
    dil5 = BatchNormalization()(dil5)

    x = Flatten()(dil5)
    x = Dense(64, activation = 'relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model([dna_input], output)

    return model

def dilatedValid4(window): #bigger window
    dna_input = Input(shape=(window, 4, 1))
    
    conv1 = Conv2D(32, kernel_size=(6,4), activation='relu', padding='valid')(dna_input)
    conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 =  MaxPooling2D((2,1),padding='same')(conv1)

    conv2 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='valid')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 =  MaxPooling2D((2,1),padding='same')(conv2)

    conv3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='valid')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 =  MaxPooling2D((4,1),padding='same')(conv3)

    conv4 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='valid')(conv3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 =  MaxPooling2D((4,1),padding='same')(conv4)

    dil1 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(2, 1))(conv4)
    dil1 = Dropout(0.2)(dil1)
    dil1 = BatchNormalization()(dil1)

    dil2 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(4, 1))(dil1)
    dil2 = Dropout(0.2)(dil2)
    dil2 = BatchNormalization()(dil2)

    x = Add()([dil1, dil2])

    dil3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(8, 1))(x)
    dil3 = Dropout(0.2)(dil3)
    dil3 = BatchNormalization()(dil3)

    x = Add()([dil1, dil2, dil3])

    dil4 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(16, 1))(x)
    dil4 = Dropout(0.2)(dil4)
    dil4 = BatchNormalization()(dil4)

    x = Add()([dil1, dil2, dil3, dil4])

    dil5 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(32, 1))(x)
    dil5 = Dropout(0.2)(dil5)
    dil5 = BatchNormalization()(dil5)

    x = Flatten()(dil5)
    x = Dense(64, activation = 'relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model([dna_input], output)

    return model

def custom(window): 
    dna_input = Input(shape=(window, 4, 1))
    
    conv1 = Conv2D(16, kernel_size=(1,4), activation='relu', padding='valid')(dna_input)

    conv2 = Conv2D(64, kernel_size=(3,1), activation='relu', padding='same')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 =  MaxPooling2D((3,1),padding='same')(conv2)

    conv3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 =  MaxPooling2D((2,1),padding='same')(conv3)

    conv4 = Conv2D(32, kernel_size=(12,1), activation='relu', padding='same')(conv3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 =  MaxPooling2D((2,1),padding='same')(conv4)

    dil1 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(2, 1))(conv4)
    dil1 = Dropout(0.2)(dil1)
    dil1 = BatchNormalization()(dil1)

    dil2 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(4, 1))(dil1)
    dil2 = Dropout(0.2)(dil2)
    dil2 = BatchNormalization()(dil2)

    x = Add()([dil1, dil2])

    dil3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(8, 1))(x)
    dil3 = Dropout(0.2)(dil3)
    dil3 = BatchNormalization()(dil3)

    x = Add()([dil1, dil2, dil3])

    dil4 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(16, 1))(x)
    dil4 = Dropout(0.2)(dil4)
    dil4 = BatchNormalization()(dil4)

    x = Add()([dil1, dil2, dil3, dil4])

    dil5 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(32, 1))(x)
    dil5 = Dropout(0.2)(dil5)
    dil5 = BatchNormalization()(dil5)

    x = Flatten()(dil5)
    x = Dense(64, activation = 'relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model([dna_input], output)

    return model

def smallcustom(window): 
    dna_input = Input(shape=(window, 4, 1))
    
    conv1 = Conv2D(16, kernel_size=(1,4), activation='relu', padding='valid')(dna_input)

    conv2 = Conv2D(64, kernel_size=(3,1), activation='relu', padding='same')(conv1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 =  MaxPooling2D((3,1),padding='same')(conv2)

    conv3 = Conv2D(64, kernel_size=(6,1), activation='relu', padding='same')(conv2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 =  MaxPooling2D((2,1),padding='same')(conv3)

    dil1 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(2, 1))(conv3)
    dil1 = Dropout(0.2)(dil1)
    dil1 = BatchNormalization()(dil1)

    dil2 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(4, 1))(dil1)
    dil2 = Dropout(0.2)(dil2)
    dil2 = BatchNormalization()(dil2)

    x = Add()([dil1, dil2])

    dil3 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(8, 1))(x)
    dil3 = Dropout(0.2)(dil3)
    dil3 = BatchNormalization()(dil3)

    x = Add()([dil1, dil2, dil3])

    dil4 = Conv2D(32, kernel_size=(6,1), activation='relu', padding='same', dilation_rate=(16, 1))(x)
    dil4 = Dropout(0.2)(dil4)
    dil4 = BatchNormalization()(dil4)

    x = Flatten()(dil4)
    x = Dense(64, activation = 'relu')(x)

    output = Dense(1, activation='sigmoid')(x)
    model = Model([dna_input], output)

    return model