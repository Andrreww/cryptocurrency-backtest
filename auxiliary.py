import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import keras
import pandas_datareader.data as web
import h5py
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = []
model.append(load_model('BTCUSDT.h5'))
model.append(load_model('ETHUSDT.h5'))
model.append(load_model('LTCUSDT.h5'))
model.append(load_model('XRPUSDT.h5'))


def transfer_data(data,normalize=True):
    col = ['open', 'high', 'low', 'close', 'volume', 'close time',
       'quote asset volume', 'number of trades', 'taker buy base asset volume',
       'taker buy quote asset volume']
    data = pd.DataFrame(data,columns = col)
    data.drop(['close time','taker buy quote asset volume','taker buy base asset volume'], 1, inplace=True)
    if normalize:        
        min_max_scaler = preprocessing.MinMaxScaler()
        data['open'] = min_max_scaler.fit_transform(data.open.values.reshape(-1,1))
        data['high'] = min_max_scaler.fit_transform(data.high.values.reshape(-1,1))
        data['low'] = min_max_scaler.fit_transform(data.low.values.reshape(-1,1))
        data['quote asset volume'] = min_max_scaler.fit_transform(data['quote asset volume'].values.reshape(-1,1))
        data['volume'] = min_max_scaler.fit_transform(data.volume.values.reshape(-1,1))
        data['number of trades'] = min_max_scaler.fit_transform(data['number of trades'].values.reshape(-1,1))
        data['close'] = min_max_scaler.fit_transform(data.close.values.reshape(-1,1))
    return np.array(data)
'''
def generate_bar(data_list):
    Function to reform format2 data into the time period you want
    Param: data_list - list containing n sequential elements from data_format2
    
    temp = np.array(data_list)
    return
    out = []
    out.append(temp[-22:,0])
    out.append(temp[-22:,1])
    out.append(temp[-22:,2])
    out.append(temp[-22:,3])
    return np.array(out)
'''
def predict(data,asset_index):
    ''' Function to reform format2 data into the time period you want
    Param: data_list - list containing n sequential elements from data_format2
    '''
    '''
    stan = data[:,-1,3]
    output = model.predict(data)
    out = []
    for i in [0,1,2,3]:
        if stan[i]<output[i]:
            out.append(1)
        else:
            out.append(-1)
    '''
    stan = np.mean(data[-1,:3])
    output = model[asset_index].predict([[data]])
    if stan*1.1<output[0,0]:
        return 1
    elif stan*0.95>output[0,0]:
        return -2
    elif stan*0.98>output[0,0]:
        return -1
    else:
        return 0