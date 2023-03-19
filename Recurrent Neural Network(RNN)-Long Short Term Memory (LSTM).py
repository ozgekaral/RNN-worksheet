# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 19:44:08 2023

@author: user202
"""

import numpy
import pandas as pd 
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

usa = pd.read_csv('USArrests.csv')
print(usa.columns)
print(usa.shape)

usa_u = usa.loc[:,['UrbanPop']].values
print(usa_u)
usa_u = usa_u.reshape(-1,1)
usa_u = usa_u.astype("float32")
print(usa_u.shape)
usa_u = usa_u.reshape(-1,1)
usa_u = usa_u.astype("float32")
print(usa_u.shape)
#MinMax
scaler = MinMaxScaler(feature_range=(0, 1))
usa_u = scaler.fit_transform(usa_u)

train_size = int(len(usa_u) * 0.20)
test_size = len(usa_u) - train_size
train = usa_u[0:train_size,:]
test = usa_u[train_size:len(usa_u),:]
print(train.shape)
print(test.shape)

time_stemp = 5
dataX = []
dataY = []
for i in range(len(train)-time_stemp-1):
    a = train[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(train[i + time_stemp, 0])
trainX = numpy.array(dataX)
trainY = numpy.array(dataY) 
print(trainX.shape)
print(trainY.shape)

dataX = []
dataY = []
for i in range(len(test)-time_stemp-1):
    a = test[i:(i+time_stemp), 0]
    dataX.append(a)
    dataY.append(test[i + time_stemp, 0])
testX = numpy.array(dataX)
testY = numpy.array(dataY) 
print(testX.shape)
print(testY.shape) 

trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX.shape)
print(testX.shape)

#Model

Sequential().add(LSTM(10, input_shape=(1, time_stemp)))
Sequential().add(Dense(1))
Sequential().compile(loss='mean_squared_error', optimizer='adam')
Sequential().fit(trainX, trainY, epochs=100, batch_size=1)

#Predict
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print(trainScore)
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print(testScore)
