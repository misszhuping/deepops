import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
# from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

# Part 1 - Data Preprocessing
# Importing the libraries
dataset_train = pd.read_csv('xxxx.csv')
training_set = dataset_train.iloc[:, 1:2].values
# print(dataset_train.head())
# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i - 60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN
# Initialising the RNN
regressor = tf.keras.Sequential()
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(tf.keras.layers.Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
regressor.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
regressor.add(tf.keras.layers.Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
regressor.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
regressor.add(tf.keras.layers.Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(tf.keras.layers.LSTM(units=50))
regressor.add(tf.keras.layers.Dropout(0.2))
# Adding the output layer
regressor.add(tf.keras.layers.Dense(units=1))
# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)

# Part 3 - Making the predictions and visualising the results
# Getting the real stock price of 2017
dataset_test = pd.read_csv('tatatest.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 76):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real TATA Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted TAT Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()