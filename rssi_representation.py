#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     rssi_representation.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2022-01-27
#
# @brief    Time series representation of RSSIs for DNN-based
#           large-scale indoor localization
#


# import cloudpickle
import numpy as np
import os
import pandas as pd


def map_rssi(x):
    y = list(zip(range(len(x)), x))
    y = np.array([a for a in y if a[1] != 100])
    if len(y) > 0: 
        return y[np.argsort(y[:,1])][::-1]
    else:
        return np.nan


def ts_length(x):
    if type(x) == np.ndarray:
        return x.shape[0]
    else:
        return 0

# training data
df_fname = 'data/ujiindoorloc/saved/training_df.h5'
if os.path.isfile(df_fname) and (os.path.getmtime(df_fname) >
                                    os.path.getmtime(__file__)):
    training_df = pd.read_hdf(df_fname)
else:
    training_df = pd.read_csv('data/ujiindoorloc/trainingdata.csv', header=0)
    training_df["RSSIs"] = training_df.iloc[:, :520].values.tolist()
    training_df["WAPs_RSSIs"] = training_df["RSSIs"].apply(map_rssi)
    training_df["TS_LENGTH"] = training_df["WAPs_RSSIs"].apply(ts_length)
    training_df.drop(training_df[training_df.TS_LENGTH <= 0].index, inplace=True)
    training_df.to_hdf(df_fname, key='training_df')

# validation data
df_fname = 'data/ujiindoorloc/saved/validation_df.h5'
if os.path.isfile(df_fname) and (os.path.getmtime(df_fname) >
                                    os.path.getmtime(__file__)):
    validation_df = pd.read_hdf(df_fname)
else:
    validation_df = pd.read_csv('data/ujiindoorloc/validationdata.csv', header=0)
    validation_df["RSSIs"] = validation_df.iloc[:, :520].values.tolist()
    validation_df["WAPs_RSSIs"] = validation_df["RSSIs"].apply(map_rssi)
    validation_df["TS_LENGTH"] = validation_df["WAPs_RSSIs"].apply(ts_length)
    validation_df.drop(validation_df[validation_df.TS_LENGTH <= 0].index, inplace=True)
    validation_df.to_hdf(df_fname, key='validation_df')

# summary of new rssi dataframes
print("Training data:")
print(training_df.head())
training_df["TS_LENGTH"].describe()
print(f'- Average number of RSSIs: {training_df["TS_LENGTH"].mean():e}')
print(f'- Maximum number of RSSIs: {training_df["TS_LENGTH"].max():e}')
print(f'- Minimum number of RSSIs: {training_df["TS_LENGTH"].min():e}')
print(f'- Number of elements without RSSIs: {training_df["WAPs_RSSIs"].isna().sum():d}')

print("Validation data:")
print(validation_df.head())
validation_df["TS_LENGTH"].describe()
print(f'- Average number of RSSIs: {validation_df["TS_LENGTH"].mean():e}')
print(f'- Maximum number of RSSIs: {validation_df["TS_LENGTH"].max():e}')
print(f'- Minimum number of RSSIs: {validation_df["TS_LENGTH"].min():e}')
print(f'- Number of elements without RSSIs: {validation_df["WAPs_RSSIs"].isna().sum():d}')
