# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 11:23:40 2021

@author: jamie
"""
#Import required modules
import pandas as pd
from datetime import datetime
import keras
import keras_preprocessing
import numpy as np
import os
import matplotlib.pyplot as plt
os.chdir("C:\Programming\Environments\dl_project_1\data")


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout, LeakyReLU
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
# import seaborn as sns
#from datetime import datetime
import keras.backend as K
import tensorflow as tf
import kerastuner as kt

#%%

df = pd.read_csv('brussels_dataset.csv')


date_time_key = 'Date'
#df.set_index('Date', inplace=True)
features = pd.DataFrame(df[['Prices', 'sunHour', 'cloudcover','humidity', 'tempC', 'windspeedKmph']])
#%% split to train, validation and test data

total_data_size = len(features)

trainval_len = round(0.85*total_data_size)


x = features
trainval_beg = 0
trainval_end = trainval_len



test_beg = trainval_len
test_end = total_data_size

trainval_data = x[trainval_beg:trainval_end]

test_data = x[test_beg:test_end]

features = trainval_data
#%%
scaler = MinMaxScaler()
scaler = scaler.fit(features)

# scaler2 = MinMaxScaler()
# scaler2 = scaler2.fit(val_data)

# scaler3 = MinMaxScaler()
# scaler3 = scaler3.fit(test_data)
#%%
features_scaled = scaler.transform(features)
features_scaled = pd.DataFrame(features_scaled)
# df2 = pd.DataFrame(df_scaled)
#%%
trainval_data_scaled = scaler.transform(trainval_data)

test_data_scaled = scaler.transform(test_data)
#%%
trainval_data_scaled = pd.DataFrame(trainval_data_scaled)
trainval_data_scaled.index = trainval_data.index

test_data_scaled = pd.DataFrame(test_data_scaled)
test_data_scaled.index = test_data.index
#%%
split_fraction = 0.8

train_split = int(split_fraction*int(trainval_data.shape[0]))

batch_size = 16
step = 1

past = 24
future = 24

#%%

start = past + future
end = start + train_split
#%%
train_data = features_scaled.loc[0 : train_split -1]
val_data = features_scaled.loc[train_split:]



#%%
x_train = train_data
y_train = features_scaled.iloc[start:end][[0]]

#%%
sequence_length = int(past/step)
#%%
trainX = []
trainY = []

n_future = 24
n_past = 24


for i in range(n_past, len(train_data) - n_future +1):
    trainX.append(train_data.iloc[i - n_past:i, 0:train_data.shape[1]])
    trainY.append(train_data.iloc[i + n_future - 1:i + n_future, 0])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
#%%
trainxx = tf.convert_to_tensor(trainX)
trainyy = tf.convert_to_tensor(trainY)

#%%
valX = []
valY = []

n_future = 24
n_past = 24


for i in range(n_past, len(val_data) - n_future +1):
    valX.append(val_data.iloc[i - n_past:i, 0:val_data.shape[1]])
    valY.append(val_data.iloc[i + n_future - 1:i + n_future, 0])

valX, valY = np.array(valX), np.array(valY)

print('valX shape == {}.'.format(valX.shape))
print('valY shape == {}.'.format(valY.shape))

valxx = tf.convert_to_tensor(valX)
valyy = tf.convert_to_tensor(valY)
#%%
#%%
x_test = test_data_scaled
y_test = test_data_scaled[[0]]

testX = []
testY = []

n_future = 24
n_past = 24


for i in range(n_past, len(test_data_scaled) - n_future +1):
    testX.append(test_data_scaled.iloc[i - n_past:i, 0:test_data_scaled.shape[1]])
    testY.append(test_data_scaled.iloc[i + n_future - 1:i + n_future, 0])

testX, testY = np.array(testX), np.array(testY)

print('testX shape == {}.'.format(testX.shape))
print('testY shape == {}.'.format(testY.shape))

testxx = tf.convert_to_tensor(testX)
testyy = tf.convert_to_tensor(testY)

#%%
x_end = len(val_data) - past - future

label_start = train_split + past + future
#%%
x_val = val_data.iloc[:x_end]
y_val = features_scaled.iloc[label_start:][[0]]
#%%
def model_builder(hp):
    # Tune the nuumber of units for first Bi-LSTM layer
    # Choose optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model_opt = Sequential()
    model_opt.add(Bidirectional(LSTM(hp_units, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]))))
    model_opt.add(Dense(1))
    
    # Tune learning rate
    # Choose an optimal value from 0.01, 0.001 or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2,1e-3,1e-4])
    
    model_opt.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='mean_absolute_error', metrics = ['mean_absolute_error']
              )

    return model_opt
#%%
my_dir = "C:\Programming\Environments\dl_project_1\data"

# tuner = kt.BayesianOptimization(model_builder,
#                       objective='val_accuracy',
#                       directory= my_dir,
#                       project_name = 'dl_energy_price2',
#                       max_trials = 10)

# tuner = kt.RandomSearch(
#     model_builder,
#     objective="val_accuracy",
#     max_trials=40,
#     project_name="hypertpoging2",
#     directory=my_dir,
# )

tuner = kt.Hyperband(model_builder,
                     objective='val_mean_absolute_error',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='dlproject3')
#%%
# Display search overview.
tuner.search_space_summary()
#%%
# Performs the hypertuning.
# tuner.search(trainX, trainY, epochs=10, validation_data=(valX,valY))

#%%
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

#%%
#model_best.fit(trainxx, trainyy, epochs=10, batch_size=16, validation_data = (valxx, valyy), verbose=1, callbacks=[es_callback, modelckpt_callback])
tuner.search(trainX, trainY, epochs=50, validation_data = (valX, valY), callbacks=[stop_early])
#%%
# get optimal hyperparameters

best_hps = tuner.get_best_hyperparameters(num_trials=30)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first Bi-LSTM
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

#%%

modeltuned = tuner.hypermodel.build(best_hps)
history = modeltuned.fit(trainX, trainY, epochs=50, validation_data=(valX,valY))

# val_mean_absolute_error_per_epoch = history.history['val_mean_absolute_error']
# best_epoch = val_mean_absolute_error_per_epoch.index(max(val_mean_absolute_error_per_epoch)) + 1
# print('Best epoch: %d' % (best_epoch,))
#%%
val_mean_absolute_error_per_epoch = history.history['val_mean_absolute_error']
best_epoch = val_mean_absolute_error_per_epoch.index(min(val_mean_absolute_error_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

#%%

modeltuned2 = tuner.hypermodel.build(best_hps)

# Retrain the model
modeltuned2.fit(trainX, trainY, epochs=best_epoch, validation_data = (valX, valY))
#%%
# Best val_mean_absolute_error So Far: 0.003508239984512329
# Total elapsed time: 00h 37m 14s

# Search: Running Trial #10

# Hyperparameter    |Value             |Best Value So Far 
# units             |64                |384               
# learning_rate     |0.001             |0.001             
# tuner/epochs      |2                 |2                 
# tuner/initial_e...|0                 |0                 
# tuner/bracket     |2                 |2                 
# tuner/round       |0                 |0      