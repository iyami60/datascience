# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:00:51 2020

@author: ISSAM
"""
#Importing libraries
import numpy as np
import pandas as pd
from keras.optimizers import SGD
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import warnings
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
#ignore all warning
warnings.simplefilter(action="ignore",category=FutureWarning)

#importing Data
data_path =  r"C:\Users\ISSAM\Documents\GitHub\data\datset\AAPL_2006-01-01_to_2018-01-01.csv"
dataset_apple = pd.read_csv(data_path)
train_set = dataset_apple.iloc[:2768,1:2]    #Take the 'Open' column from data
test_set =  dataset_apple.iloc[2768:,1:2]    #Take the 'Open' column from data

#scaling Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))              #mettre les valeur entre 0 et 1
training_set_scaled = scaler.fit_transform(train_set)
plt.plot(training_set_scaled, color='red')
plt.title('NOrmalisation du données')
plt.xlabel('Time')
plt.legend()
plt.show()

#Make data transformation like 60/70 time steps and 1 Indicators.
X_train,y_test = [],[]
x = 80      #time step.
last_row = training_set_scaled.shape[0]
for i in range(x,last_row):
    X_train.append(training_set_scaled[i-x:i,0])
    y_test.append(training_set_scaled[i,0])
X_train = np.array(X_train)
y_test = np.array(y_test)

#reshapeDta
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))  


"""
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
regressor.fit(X_train,y_test,batch_size = 40,epochs = 30)
"""
#save regressor
model = Sequential()
model.add(SimpleRNN(32))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # fit the RNN model
model.fit(X_train, y_test, epochs=100, batch_size=150)

#GRU_predicted_stock_price = regressorGRU.predict(X_test)
#GRU_predicted_stock_price = sc.inverse_transform(GRU_predicted_stock_price)

model.save("Simplernn100")

#Get teh real price of stock of the year that we want.
name1 = r"C:\Users\ISSAM\Documents\GitHub\data\datset\AAPL_2006-01-01_to_2018-01-01.csv"
dataset_test = pd.read_csv(data_path)
real_price_stock = dataset_test.iloc[2768:,1:2].values

#concatenate dataset
dataset_total = pd.concat((train_set['Open'],dataset_test['Open']),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-80:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.fit_transform(inputs)


model = tf.keras.models.load_model(r'C:\Users\ISSAM\Documents\GitHub\data\model_stafe_1')

X_test = []
for i in range(80,331):
     X_test.append(inputs[i-80:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

#visualisation result
plt.plot(real_price_stock, color='red', label='Real APLL stock price')
plt.plot(predicted_stock_price, color='blue', label='Predicted AAPLstock price')
plt.title('AAPLL Stock Price prediction')
plt.xlabel('Time')
plt.ylabel('Ggl stock price')
plt.legend()
plt.show()
