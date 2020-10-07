# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:52:55 2020

@author: ISSAM
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


data_path = r"C:\Users\ISSAM\Documents\GitHub\data\datset\AAPL_2006-01-01_to_2018-01-01.csv"
dataset_apple = pd.read_csv(data_path)
dataset_apple_train = dataset_apple.iloc[:2768]
dataset_apple_test  = dataset_apple.iloc[2768:]
real_price_stock = dataset_apple.iloc[2768:,1:2].values
#concatenate dataset
dataset_total = pd.concat((dataset_apple_train['Open'],dataset_apple_test['Open']),axis=0)
dataset = dataset_total[len(dataset_apple_train)-len(dataset_apple_test)-80:].values
inputs = dataset.reshape(-1,1)


scaler = MinMaxScaler(feature_range=(0, 1))
inputs = scaler.fit_transform(inputs)

X_test = []
for i in range(80,331):
     X_test.append(inputs[i-80:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
#load_Model
new_model = tf.keras.models.load_model(r'C:\Users\ISSAM\Documents\GitHub\data\Stage_2A\single_layer_rnn')

predicted_stock_price = new_model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

#visualisation result
plt.plot(real_price_stock, color='red', label='Real APLL stock price')
plt.plot(predicted_stock_price, color='blue', label='Predicted AAPLstock price')
plt.title('AAPLL Stock Price prediction')
plt.xlabel('Time')
plt.ylabel('APPLE stock price')
plt.legend()
plt.show()