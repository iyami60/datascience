# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:40:56 2020

@author: ISSAM
"""
#PART1: Data Processing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training set
name = r"C:\Users\ISSAM\Documents\GitHub\data\Google_Stock_Price_Train.csv"
dataset_train = pd.read_csv(name)
training_set = dataset_train.iloc[:,1:2]
training_set = training_set.values

#Data (Features) Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = scaler.fit_transform(training_set)

#Creating DATA STRUCTURE with x time steps and 1 output
X_train,y_train = [],[]
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train),np.array(y_train)
#reshaping
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))   

#PART2: Building RNN
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

#initialize a RNN 
regressor = Sequential()
#first LSTM LAYERS ( units, retuen_seqeunces, inputèshape , indicators) 
#Dropout Regularization
regressor.add(LSTM(units = 50,return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2)) #means we lose 20% of neurons at eacch iteration

#The second LSTM LAYERS
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

#The third LSTM LAYERS
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

#The Fourth LSTM LAYERS
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#adding the output layer
regressor.add(Dense(units = 1))
#compiling regressor
regressor.compile(optimizer = 'adam', loss = "mean_squared_error")
#fitting regressor
regressor.fit(X_train, y_train, epochs=100, batch_size = 32)





