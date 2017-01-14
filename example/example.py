import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def main():

	# Read data
	data = pd.read_csv('data.csv')
	data = data.set_index('DATETIME')

	# re-classify
	data['x'] = data['CLOSE'].pct_change()
	data['x'] = (data['x']-pd.rolling_min(data['x'],window=30))/(pd.rolling_max(data['x'],window=30)-pd.rolling_min(data['x'],window=30))
	data['y'] = np.zeros(data['CLOSE'].shape)
	data['y'].loc[(data['CLOSE'].shift(-5)-data['CLOSE']) > 0] = 1
	data['y'].loc[(data['CLOSE'].shift(-5)-data['CLOSE']) < 0] = 0

	data = data.dropna()
	data = data[['x','y']]
	print data.tail(10)

	# Format
	y = data['y'].as_matrix()
	x = data.drop('y',1).as_matrix()

	# split into train and test sets
	train_size = int(len(data) * 0.67)
	test_size = len(data) - train_size
	x_train, x_test = x[0:train_size,:], x[train_size:len(data),:]
	y_train, y_test = y[0:train_size], y[train_size:len(data)]

	# reshape input to be [samples, time steps, features]
	x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
	x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

	# model

	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_dim=1))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(x_train, y_train, nb_epoch=5, batch_size=1, verbose=2)

	# make predictions
	trainPredict = model.predict(x_train)
	testPredict = model.predict(x_test)

	

if __name__ == "__main__":

  print("Deep Learning Implementation.")

  try:

    main()

  except KeyboardInterrupt:
    print('Interupted...Exiting...')
