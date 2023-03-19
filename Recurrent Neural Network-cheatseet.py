# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 17:58:03 2023

@author: user202
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout

price_train = pd.read_csv('train.csv')
print(price_train.columns)

p_train=price_train.loc[:, ['LotArea']].values
print(p_train)
#MinMax 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
train_scaled = scaler.fit_transform(p_train)
train_scaled
plt.plot(train_scaled)
plt.show()
print(train_scaled.size)
#Like train_test_split
X_train = []
y_train = []
timesteps = 80
for i in range(timesteps, 1460):
    X_train.append(train_scaled[i-timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.size)
print(y_train.size)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.size)
#MODEL
Sequential().add(SimpleRNN(units = 30,activation='tanh', return_sequences = True, input_shape = (X_train.shape[1], 1)))
Sequential().add(Dropout(0.3))

# Adding a second RNN layer and some Dropout regularisation
Sequential().add(SimpleRNN(units = 30,activation='tanh', return_sequences = True))
Sequential().add(Dropout(0.3))

# Adding a third RNN layer and some Dropout regularisation
Sequential().add(SimpleRNN(units = 30,activation='tanh', return_sequences = True))
Sequential().add(Dropout(0.3))

# Adding a fourth RNN layer and some Dropout regularisation
Sequential().add(SimpleRNN(units = 30))
Sequential().add(Dropout(0.3))

# Adding the output layer
Sequential().add(Dense(units = 1))

# Compiling the RNN
Sequential().compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
Sequential().fit(X_train, y_train, epochs = 250, batch_size = 32)

#Predict
price_test = pd.read_csv('test.csv')
print(price_test.size)

p_test=price_test.loc[:, ['LotArea']].values
print(p_test)

total = pd.concat((price_train['Open'], price_test['Open']), axis = 0)
inputs = total[len(total) - len(price_test) - timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs)
inputs

X_test = []
for i in range(timesteps, 70):
    X_test.append(inputs[i-timesteps:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted = Sequential().predict(X_test)
predicted = scaler.inverse_transform(predicted)
print(predicted)


