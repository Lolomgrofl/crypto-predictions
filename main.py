import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

# Desired crypto
crypto = 'ETH'
curr = 'EUR'

# Period of time for analysis
start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

# Loading dataset from yahoo-finance
dataset = web.DataReader(f'{crypto}-{curr}', 'yahoo', start, end)
print(dataset.head())

# Data preprocessing

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset['Close'].values.reshape(-1, 1))

prediction_days = 60

X_train, y_train = [], []

for i in range(prediction_days, len(scaled_data)):
    X_train.append(scaled_data[i - prediction_days:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building neural network

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # price prediction of crypto

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=15, batch_size=32)


# Testing the models

test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()
test_dataset = web.DataReader(f'{crypto}-{curr}', 'yahoo', test_start, test_end)
actual_prices = test_dataset['Close'].values

final_dataset = pd.concat((dataset['Close'], test_dataset['Close']), axis=0)

model_inputs = final_dataset[len(final_dataset) - len(test_dataset) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

X_test = []

for i in range(prediction_days, len(model_inputs)):
    X_test.append(model_inputs[i - prediction_days: i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

prediction_prices = model.predict(X_test)
prediction_prices = scaler.inverse_transform(prediction_prices)


# Visualizing
plt.plot(actual_prices, color='blue', label='Actual Prices')
plt.plot(prediction_prices, color='red', label='Prediction Prices')
plt.title(f'{crypto} price prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()

# Predict Next Day

real_data = [model_inputs[len(model_inputs)+1 - prediction_days: len(model_inputs) + 1, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict_proba(real_data)
prediction = scaler.inverse_transform(prediction)
print(prediction.round(2))

