import pandas as pd
import numpy as np
from datetime import datetime
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def build_model(shape):
    model = Sequential()
    model.add(LSTM(4, input_shape=shape))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model



y = [1483228800, 1514764800, 1546300800, 1577836800, 9999999999]

for year in range(len(y)-1):
    print(y[year])
    for curr in ['BTC', 'ETH', 'LTC', 'XRP']:
        print(curr)
        if 'data' not in locals():
            response = requests.get(
                "https://poloniex.com/public?command=returnChartData&currencyPair=USDT_{}&start={}&end={}&period=1800".format(curr, y[year], y[year+1]))
            data = pd.DataFrame(response.json())
            data.date = pd.to_datetime(data.date.apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
            data = data.set_index('date')
            data = data[['weightedAverage', 'volume']]
        else:
            response = requests.get(
                "https://poloniex.com/public?command=returnChartData&currencyPair=USDT_{}&start={}&end={}&period=1800".format(curr, y[year], y[year+1]))
            data2 = pd.DataFrame(response.json())
            data2.date = pd.to_datetime(data2.date.apply(lambda x: datetime.utcfromtimestamp(x).strftime('%Y-%m-%d %H:%M:%S')))
            data2 = data2.set_index('date')
            data = data.merge(data2[['weightedAverage', 'volume']], left_index=True, right_index=True, suffixes=['', '_'+curr])
    if 'data_' not in locals():
        data_ = data.copy()
    else:
        data_ = data_.append(data.copy())
    del data

scaler = MinMaxScaler(feature_range=(0, 1))
scaler_target = MinMaxScaler(feature_range=(0, 1))

test_train_split = .8
target = data_['weightedAverage'].ffill()[1:].values
# data_[['weightedAverage', 'volume']] = data_[['weightedAverage', 'volume']].shift(1)
data_ = data_.shift(1)
window_size = 5
data_ = data_.ffill().dropna()
target = target.reshape(-1, 1)



data_[:int(len(data_)*test_train_split)] = scaler.fit_transform(data_[:int(len(data_)*test_train_split)])
data_[int(len(data_)*test_train_split):] = scaler.transform(data_[int(len(data_)*test_train_split):])

target[:int(len(target)*test_train_split)] = scaler_target.fit_transform(target[:int(len(target)*test_train_split)])
target[int(len(target)*test_train_split):] = scaler_target.transform(target[int(len(target)*test_train_split):])


data_ = data_.values
nrows = data_.shape[0] - window_size + 1
p, q = data_.shape
m, n = data_.strides
strided = np.lib.stride_tricks.as_strided
out = strided(data_, shape=(nrows, window_size, q), strides=(m, m, n))
out = out[1:]

target = target[window_size:]

trainX = out[:int(len(out)*test_train_split)]
trainY = target[:int(len(out)*test_train_split)]
testX = out[int(len(out)*test_train_split):]
testY = target[int(len(out)*test_train_split):]

model = build_model(out[0].shape)
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
