#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:16 2022
@author: lou
"""    
 
from tensorflow.keras.layers import ReLU, BatchNormalization, Add
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
 
def ConvBlock(input, filters, kernel):
   current = Conv2D(filters, (kernel, input.shape[2]))(input)
   current = BatchNormalization()(current)
   current = ReLU()(current)
   current = Dropout(0.2)(current)
   output = MaxPooling2D((2,1))(current)
   return output
 
def DenseDilatedConvBlock(input, filters, kernel, dilation_rate):
   current = Conv2D(filters//2, (kernel, 1), padding='same', dilation_rate=(dilation_rate,1))(input)
   current = BatchNormalization()(current)
   current = ReLU()(current)
   current = Conv2D(filters, (1,1))(current)
   current = BatchNormalization()(current)
   current = Dropout(0.2)(current)
   output = Add()([input, current])
   return output