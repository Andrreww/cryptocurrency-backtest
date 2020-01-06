# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:05:01 2019

@author: Shihao
"""
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

from auxiliary import transfer_data, predict  # auxiliary is a local py file containing some functions


# Here you can
# 1. import necessary python packages for your strategy
# 2. Load your own facility files containing functions, trained models, extra data, etc for later use
# 3. Set some global constants
# Note:
# 1. You should put your facility files in the same folder as this strategy.py file
# 2. When load files, ALWAYS use relative path such as "data/facility.pickle"
# DO NOT use absolute path such as "C:/Users/Peter/Documents/project/data/facility.pickle"
  # only consider BTC (the **second** crypto currency in dataset)
bar_length = 30  # Number of minutes to generate next new bar
# Here is your main strategy function
# Note:
# 1. DO NOT modify the function parameters (time, data, etc.)
# 2. The strategy function AWAYS returns two things - position and memory:
# 2.1 position is a np.array (length 4) indicating your desired position of four crypto currencies next minute
# 2.2 memory is a class containing the information you want to save currently for future use


def handle_bar(counter,  # a counter for number of minute bars that have already been tested
               time,  # current time in string format such as "2018-07-30 00:30:00"
               data,  # data for current minute bar (in format 2)
               init_cash,  # your initial cash, a constant
               transaction,  # transaction ratio, a constant
               cash_balance,  # your cash balance at current minute
               crypto_balance,  # your crpyto currency balance at current minute
               total_balance,  # your total balance at current minute
               position_current,  # your position for 4 crypto currencies at this minute
               memory  # a class, containing the information you saved so far
               ):
    # Here you should explain the idea of your strategy briefly in the form of Python comment.
    # You can also attach facility files such as text & image & table in your team folder to illustrate your idea
    position_new = position_current
    
    if counter == 0:
        memory.data_list = []
        memory.history = []
        memory.weight = [0,0,1,0]
        memory.barcount = 0
    memory.data_list.append(data)
    # The idea of my strategy:
    # Buy 10 BTC at the very beginning and hold it to the end.
    if (counter + 1) % bar_length == 0:
        memory.barcount += 1
        k = np.array(memory.data_list)
        his = np.array(memory.history)
        prebar = []
        for asset_index in range(4):
            temp = k[-22:,asset_index,]
            bar = transfer_data(temp)
            buy = predict(bar,asset_index)
            prebar.append(buy)
            flag = 0
            if position_new[asset_index] == 0 and memory.barcount>=5:
                for i in range(1):
                    if his[-i-1,asset_index]<=0:
                        flag = 1
            if flag == 1:
                continue
            if buy==-2 :
                if memory.barcount>=5:
                    for i in range(2):
                        if his[-i-1,asset_index]>=0:
                            flag = 1
                            position_new[asset_index] = position_new[asset_index] * 0.2
                    if flag == 0:
                        position_new[asset_index] = 0
            elif buy == 1:
    #            position_new[asset_index] += cash_balance/(4*np.mean(temp[asset_index][-1,:3]))
    #            if np.mean(temp[-1,:3])<init_cash/4:
    #                position_new[asset_index] += 1
                if position_new[asset_index]== 0:
                    position_new[asset_index] = init_cash*0.8*memory.weight[asset_index]/(2*temp[-1,3])
                else:
                    position_new[asset_index] = position_new[asset_index] + 0.5 * (init_cash*0.8*memory.weight[asset_index]/temp[-1,3] - position_new[asset_index])
            elif buy == -1:
    #            position_new[asset_index]  = position_new[asset_index] *0.5
                position_new[asset_index] = position_new[asset_index] * 0.5
    #            position_new[asset_index] = np.max(position_new[asset_index],0)
        memory.history.append(prebar)
        print(position_new)
    return position_new, memory