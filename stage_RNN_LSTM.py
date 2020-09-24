# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:00:51 2020

@author: ISSAM
"""
#Importing libraries
from keras.models import Sequential
from keras.layers import LSTM, Dense,Dropout
import numpy as np
import pandas as pd
import matplotlib as plt
import warnings
#ignore all warning
warnings.simplefilter(action="ignore",category=FutureWarning)

#importing Data
data_path = r"C:\Users\ISSAM\Documents\GitHub\data\datset\AAPL_2006-01-01_to_2018-01-01.csv"
dataset_apple = pd.read_csv(data_path)
train_set = dataset_apple.iloc[:2768,1:2].values #Take the 'Open' column from data
test_set =  dataset_apple.iloc[2768:,1:2].values

#scaling Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))     #mettre les valeur entre 0 et 1
training_set_scaled = scaler.fit_transform(train_set)

#Make data transformation like 60/70 time steps and 1 Indicators.
X_train,y_test = [],[]
x = 80       #time step.
last_row = training_set_scaled.shape[0]
for i in range(x,last_row):
    X_train.append(training_set_scaled[i-x:i,0])
    y_test.append(training_set_scaled[i,0])
X_train = np.array(X_train)
y_test = np.array(y_test)

#reshapeDta
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))  

#STEP1: initialize the Model
regressor = Sequential()
#Step2: make building layers
#FisrtLayers.(LSTM)
regressor.add(LSTM(units = 50,return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

#second layers (LSTM)
regressor.add(LSTM(units = 50,return_sequences = True))
regressor.add(Dropout(0.2))

#third Layers. (LSTM)
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

#Thr flast layers (LSTM)
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#add the Output layer
regressor.add(Dense(units = 1))

#add the compiler
regressor.compile(optimizer = 'Adam',loss = 'mean_squared_error')

#fit regressor with DATA
regressor.fit(X_train,y_test,batch_size = 40,epochs = 60)

#save regressor

