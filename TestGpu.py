# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:31:09 2019

@author: Shihao
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:46:41 2019

@author: Shihao
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import load_model
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os
import tensorflow as tf

#进行配置，每个GPU使用60%上限现存
#from keras import backend as K
#config = tf.compat.v1.ConfigProto()
 
#config.gpu_options.allow_growth=True
 
#sess = tf.compat.v1.Session(config=config)
 
#K.set_session(sess)

#config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
#sess = tf.compat.v1.Session()(config=config) 
#keras.backend.set_session(sess)


from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn import svm
import statsmodels.api as sm


X = sm.add_constant(svmtrain) # adding a constant
Y = list(d)
model = sm.OLS(Y, X).fit()
model.summary()

Liner.fit(svmtrain[:8000],)
Liner_out = Liner.predict(svmtrain[8000:])
print(np.linalg.norm(Liner_out-int(d[8000:]/svmtrain[8000:]['close'])))

svmtrain = df[:10001]
d = svmtrain['close'][1:]
svmtrain = svmtrain[:-1]

svr_rbf=svm.SVR(kernel='rbf',C=1e3,gamma=0.1)
svr_rbf.fit(svmtrain[:8000],d[:8000])
svr_rbf_out = svr_rbf.predict(svmtrain[8000:])
print(np.linalg.norm(svr_rbf_out-d[8000:]))

svr_lin=svm.SVR(kernel='linear',C=1e3)
svr_lin.fit(svmtrain[:8000],d[:8000])
svr_lin_out = svr_lin.predict(svmtrain[8000:])
print(np.linalg.norm(svr_lin_out-d[8000:]))

svr_poly=svm.SVR(kernel='poly',C=1e3,degree=2)
svr_poly.fit(svmtrain[:8000],d[:8000])
svr_poly_out = svr_poly.predict(svmtrain[8000:])
print(np.linalg.norm(svr_poly_out-d[8000:]))

import xgboost as xgb
xgbr=xgb.XGBRegressor()
xgbr.fit(svmtrain[:8000],d[:8000])#拟合
xgbr_out =xgbr.predict(svmtrain[8000:])#预测
plt.figure(figsize = (10,4))
plt.bar(svmtrain.columns,xgbr.feature_importances_)
plt.title('feature_importances with XGB')
print(np.linalg.norm(xgbr_out-d[8000:]))


from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor()
clf.fit(svmtrain[:8000],d[:8000])
plt.figure(figsize = (10,4))
plt.bar(svmtrain.columns,clf.feature_importances_)
plt.title('feature_importances with RF')
rf_out =clf.predict(svmtrain[8000:])
print(np.linalg.norm(rf_out-d[8000:]))


lstm_out = model.predict(X_train[46000:50000])
print(np.linalg.norm(lstm_out-y_train[46000:50000]))









seq_len = 22
bar = 30
d = 0.2
shape = [7, seq_len, 1] # feature, window, output
neurons = [128, 128, 32, 1]
epochs = 30

def get_stock_data(normalize=True):
    df = pd.read_csv('preprocessed_BTCUSDT.csv')
    df.drop(['close time','taker buy quote asset volume','taker buy base asset volume'], 1, inplace=True)
    if normalize:        
        min_max_scaler = preprocessing.MinMaxScaler()
        df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
        df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
        df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
        df['quote asset volume'] = min_max_scaler.fit_transform(df['quote asset volume'].values.reshape(-1,1))
        df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))
        df['number of trades'] = min_max_scaler.fit_transform(df['number of trades'].values.reshape(-1,1))
        df['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
        df = df.set_index('time')
    return df
df = get_stock_data(normalize=True)

def get_stock_data(normalize=True):
    df = pd.read_csv('preprocessed_BTCUSDT.csv')
    df.drop(['close time'], 1, inplace=True)
    if normalize:        
        min_max_scaler = preprocessing.MinMaxScaler()
        df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
        df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
        df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
        df['quote asset volume'] = min_max_scaler.fit_transform(df['quote asset volume'].values.reshape(-1,1))
        df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))
        df['number of trades'] = min_max_scaler.fit_transform(df['number of trades'].values.reshape(-1,1))
        df['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
        df['taker buy quote asset volume'] = min_max_scaler.fit_transform(df['taker buy quote asset volume'].values.reshape(-1,1))
        df['taker buy base asset volume'] = min_max_scaler.fit_transform(df['taker buy base asset volume'].values.reshape(-1,1))
        df = df.set_index('time')
        
    return df
df = get_stock_data(normalize=True)





def load_data(stock, seq_len, bar):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() 
    sequence_length = seq_len  # index starting from 0
    result = []
                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    for index in range(len(data) - sequence_length - bar): # maxmimum date = lastest date - sequence length
        result.append(data[index: index + sequence_length + bar]) # index : index + 22days
    
    result = np.array(result)
    row = round(0.9 * result.shape[0]) # 90% split
    
    train = result[:int(row), :] # 90% date
    X_train = train[:, :sequence_length] # all data until day m
#    y_train = train[:, -1][:,3] # day m + 1 adjusted close price
    d = train[:, sequence_length:,:4]
    y_train = np.mean(np.mean(d,axis = 1),axis = 1)
    

    X_test = result[int(row):, :sequence_length]
    #y_test = result[int(row):, -1][:,3]
    d = result[int(row):, sequence_length:,:4]
    y_test = np.mean(np.mean(d,axis = 1),axis = 1)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  
 
    return [X_train, y_train, X_test, y_test]
 
X_train, y_train, X_test, y_test = load_data(df, seq_len,bar)


def build_model2(layers, neurons, d):
    model = Sequential()
    
    model.add(LSTM(neurons[0], input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
        
    model.add(LSTM(neurons[1], input_shape=(layers[1], layers[2]), return_sequences=False))
    model.add(Dropout(d))
        
    model.add(Dense(neurons[2],kernel_initializer="uniform",activation='relu'))        
    model.add(Dense(neurons[3],kernel_initializer="uniform",activation='linear'))
    # model = load_model('my_LSTM_stock_model1000.h5')
    adam = keras.optimizers.Adam(decay=0.2)
    model.compile(loss='mse',optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
 
model = build_model2(shape, neurons, d)

history=model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=epochs,
    validation_split=0.1,
    verbose=1)

model = load_model('C:\\Users\\Shihao\\Desktop\\Courses\\Attachments\\MAFS5140-20190930T085423Z-001\\MAFS5140\\python\\Drunken man4\\Drunken man4\\BTCUSDT.h5')


import matplotlib.pyplot as plt
def percentage_difference(model, X_test, y_test):
    percentage_diff=[]
 
    p = model.predict(X_test)
    for u in range(len(y_test)): # for each data index in test data
        pr = p[u][0] # pr = prediction on day u
 
        percentage_diff.append((pr-y_test[u]/pr)*100)
    return p,percentage_diff

p,percentage_diff = percentage_difference(model, X_train[46000:50000], y_train[46000:50000])

def denormalize(normalized_value):
    
    df = pd.read_csv('preprocessed_BTCUSDT.csv')
    df = df['close'].values.reshape(-1,1)
    normalized_value = normalized_value.reshape(-1,1)
    #return df.shape, p.shape
    min_max_scaler = preprocessing.MinMaxScaler()
    a = min_max_scaler.fit_transform(df)
    new = min_max_scaler.inverse_transform(normalized_value)
    return new


def plot_result(newp, newy_test):
    newp = denormalize(newp)
    newy_test = denormalize(newy_test)
    plt.plot(newp, color='red', label='Prediction')
    plt.plot(newy_test,color='blue', label='Actual')
    plt.legend(loc='best')
    plt.xlabel('minutes')
    plt.ylabel('Close')
    plt.show()

plot_result(p,y_train[46000:50000])
model.save('BTCUSDT.h5')

