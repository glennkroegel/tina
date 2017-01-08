#!/usr/bin/env python

import numpy as np
import pandas as pd
import talib as ta
import copy
import cPickle as pickle
from calculations import *

from sklearn import linear_model, cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier


class Model(object):
	"""docstring for ClassName"""
	def __init__(self, data, options):
		# Constructor
		self.options = options
		self.raw_data = self.getData(data)
		self.y = self.binaryClassification()
		self.x = self.features()
		self.x, self.y = self.prepareData()
		
		self.model = None
		self.score = None
		
	def getData(self, data):
		# load pricing data
		self.raw_data = pd.read_csv(data)
		self.raw_data = self.raw_data.set_index("DATETIME")
		return self.raw_data

	def features(self):
		# raw feature calculation - done first before filters
		x = copy.deepcopy(self.raw_data)
		x['WILLR'] = taCalcIndicator(x, 'WILLR', window = 30)
		x['WILLR_D1'] = x['WILLR'].pct_change()
		x['WILLR_D2'] = x['WILLR'].pct_change(2)
		x['WILLR_D5'] = x['WILLR'].pct_change(5)
		#x = x.drop('WILLR',1)

		x['RSI'] = taCalcIndicator(x, 'RSI', window = 30)
		'''x['RSI_D1'] = x['RSI'].diff()
		x['RSI_D2'] = x['RSI'].diff(2)
		x = x.drop('RSI',1)'''

		#x['STOCH'] = taCalcIndicator(x, 'CCI', window = 30)

		#x['BOP'] = taCalcIndicator(x, 'BOP')

		# jarque-bera
		#x = pd.concat([x, jarque_bera(x)], axis=1)

		# breakout-points
		#x = pd.concat([x, breakout_points(x, 30, 2)], axis=1)

		# ribbon - sma
		#x = pd.concat([x, distance_metric(ribbon_sma(x))], axis=1)
		#x = pd.concat([x, width_metric(ribbon_sma(x))], axis=1)

		# ribbon - willr
		#x = pd.concat([x, distance_metric(ribbon_willr(x), prefix='willr_hamming')], axis=1)

		return x

	def binaryClassification(self):

		period = self.options['time_period']
		prices = copy.deepcopy(self.raw_data)
		if self.options['classification_method'] == 'on_close':
			prices['y'] = np.zeros(prices['CLOSE'].shape)
			prices['NEXT'] = prices['CLOSE'].shift(-period)
			prices['Diff'] = prices['NEXT'] - prices['CLOSE']

			prices['y'].loc[prices['Diff'] > 0] = 1
			prices['y'].loc[prices['Diff'] < 0] = 0
			prices['y'].loc[prices['Diff'] == 0] = 2
			return prices['y']
		

	def prepareData(self):

		x = copy.deepcopy(self.x)
		y = copy.deepcopy(self.y)
		temp = pd.concat([x, y], axis = 1)
		temp = temp.replace([np.inf, -np.inf], np.nan)
		temp = temp.dropna()
		temp = temp.loc[temp['y'] != 2]

		try:
			temp.index = pd.to_datetime(temp.index, format = "%d/%m/%Y %H:%M")
		except:
			temp.index = pd.to_datetime(temp.index, format = "%Y-%m-%d %H:%M:%S")

		temp = temp.between_time(dt.time(self.options['hour_start'],00), dt.time(self.options['hour_end'],00))
	
		x_prepared = temp.drop('y',1)
		y_prepared = temp[['y']]

		assert(len(x_prepared) == len(y_prepared))
		assert(x_prepared.index == y_prepared.index).all()

		return x_prepared, y_prepared

	def train_model(self):
		# place split & train here
		# execute in main
		# allows child of class without training 
		# for forward test
		self.X_train, self.X_test, self.Y_train, self.Y_test = self.split()
		self.model = self.train()
		self.score = self.evaluate()

	def split(self):

		price_info_cols = list(self.raw_data.columns.values)

		x = self.x.drop(price_info_cols,1).as_matrix()
		y = self.y.values.ravel()
		assert(len(x) == len(y))
		x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, train_size = self.options['split'], random_state = 42)

		return x_train, x_test, y_train, y_test


	def train(self):
		# train model
		ls_x = self.X_train
		ls_y = self.Y_train
		clf = RandomForestClassifier(n_estimators = 1000, criterion = 'gini', min_samples_leaf = 250, n_jobs = -1,
																 random_state = 62, class_weight = 'balanced', bootstrap = False)
		assert not np.any(np.isnan(ls_x) | np.isinf(ls_x))
		clf.fit(ls_x, ls_y)
		return clf

	def evaluate(self):
		# evaluate model
		assert self.model is not None
		px = self.model.predict_proba(self.X_test)
		score = self.model.score(self.X_test, self.Y_test)
		return score

	def export(self):
		# export model
		with open('model.pkl', 'wb') as model:
			pickle.dump(self.model, model)

	def forwardTest(self, data):
		child = Model(data, self.options)
		price_info_cols = list(child.raw_data.columns.values)
		x = child.x.drop(price_info_cols,1).as_matrix()
		y = child.y.values.ravel()
		px = self.model.predict_proba(x)
		child.score = self.model.score(x, y)
		print child.score

		# Result formatting for backtest
		df_result = pd.DataFrame(zip(y,px[:,1]))
		df_result.to_csv('forward_test.csv')

def main():

	options = {'time_period': 5,
							'split': 0.7,
						 'classification_method': 'on_close',
						 	'hour_start': 21,
						 	'hour_end': 23}

	my_model = Model('EURGBP1_201213.csv', options)
	#print my_model.x.tail(10)
	my_model.x.to_csv('feature_vector.csv')
	my_model.train_model()
	print my_model.score
	my_model.export()
	#my_model.forwardTest('EURGBP1_201415.csv')

if __name__ == "__main__":

  print("Running")

  try:

    main()

  except KeyboardInterrupt:

    print('Interupted...Exiting...')
