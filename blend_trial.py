#!/opt/conda/bin/python

import numpy as np
import pandas as pd
import talib as ta
import copy
import cPickle as pickle
from calculations import *

from sklearn import linear_model, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler, MinMaxScaler, LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from talib import abstract

class Model(object):
	"""docstring for ClassName"""
	def __init__(self, data, options):
		# Constructor
		self.options = options
		self.raw_data = self.getData(data)
		self.y = self.binaryClassification()
		self.x = self.features()
		self.x, self.y = self.prepareData()
		self.feature_list = [col for col in self.x if col not in self.raw_data.columns]
		
		self.model = None
		self.scaler = None
		self.predictions = None
		self.px = None
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
		x['WILLR_D1'] = x['WILLR'].diff()
		#x['WILLR_D2'] = x['WILLR'].diff(2)
		#x['WILLR_D5'] = x['WILLR'].diff(5)

		#x['ADOSC'] = taCalcIndicator(x, 'ADOSC', window = 30)
		#x['ADOSC_D1'] = x['ADOSC'].diff()

		#x['ULTOSC'] = taCalcIndicator(x, 'ULTOSC', window = 30)
		#x['ULTOSC_D1'] = x['ULTOSC'].diff()

		#x['RSI'] = taCalcIndicator(x, 'RSI', window = 30)
		#x['RSI_D1'] = x['RSI'].diff()
		#x['RSI_D2'] = x['RSI'].diff(2)
		#x['RSI_D5'] = x['RSI'].diff(5)

		#x['CCI'] = taCalcIndicator(x, 'CCI', window = 30)
		#x['CCI_D1'] = x['CCI'].diff()

		#x['BOP'] = taCalcIndicator(x, 'BOP')
		#x['dBOP'] = x['BOP'].diff()

		#x['ATR'] = taCalcIndicator(x, 'ATR', window = 14)
		#x['dATR'] = x['ATR'].diff()

		x['ADX'] = taCalcIndicator(x, 'ADX', window = 14)
		x['dADX'] = x['ADX'].diff()

		#x['ROC'] = taCalcIndicator(x, 'ROC')
		#x['sigma'] = x['CLOSE'].rolling(window = 30, center = False).std()
		#x['dsigma'] = x['sigma'].diff()
	
		#x['BP30'] = breakawayEvent(x, window =30)
		'''x['BP31'] = breakawayEvent(x, window =31)
		x['BP32'] = breakawayEvent(x, window =32)
		x['BP33'] = breakawayEvent(x, window =33)
		x['BP34'] = breakawayEvent(x, window =34)
		x['BP35'] = breakawayEvent(x, window =35)
		x['BP36'] = breakawayEvent(x, window =36)
		x['BP40'] = breakawayEvent(x, window =40)
		x['BP60'] = breakawayEvent(x, window =60)
		x['BP120'] = breakawayEvent(x, window =120)'''
		
		# jarque-bera
		#x = pd.concat([x, jarque_bera(x)], axis=1)

		# breakout-points
		#x = pd.concat([x, breakout_points(x, 30, 2)], axis=1)

		# ribbon - sma
		#x = pd.concat([x, width_metric(ribbon_sma(x))], axis=1)
		#x = pd.concat([x, distance_metric(ribbon_sma(x))], axis=1)

		# ribbon - willr
		#x = pd.concat([x, distance_metric(ribbon_willr(x), prefix='willr_hamming')], axis=1)

		# Hour dummies
		'''try:
			x.index = pd.to_datetime(x.index, format='%d/%m/%Y %H:%M')
  		except:
  			x.index = pd.to_datetime(x.index, format='%Y-%m-%d %H:%M:%S')
		
		#x = pd.concat([x, hour_dummies(x)], axis=1)
		x['hour'] = x.index.hour/100'''


		#print x.tail(10)

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
		# Filter
		#temp = temp.loc[x['BP30'] != 0]
		#temp = temp.loc[x['ADX']>25]

		'''try:
			temp.index = pd.to_datetime(temp.index, format = "%d/%m/%Y %H:%M")
		except:
			temp.index = pd.to_datetime(temp.index, format = "%Y-%m-%d %H:%M:%S")

		temp = temp.between_time(dt.time(self.options['hour_start'],00), dt.time(self.options['hour_end'],00))'''

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

		if self.options['scale'] == True:
			self.X_train, self.X_test = self.scale(self.X_train, self.X_test)

		print self.X_test[1000]
		self.model = self.train()
		self.score = self.evaluate()

	def split(self):

		price_info_cols = list(self.raw_data.columns.values)

		x = self.x.drop(price_info_cols,1).as_matrix()
		y = self.y.values.ravel()
		assert(len(x) == len(y))

		#x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, train_size = self.options['split'], random_state = 42)

		p = 0.8
		ix_train = int(p*len(x))
		x_train = x[0:ix_train]
		x_test = x[ix_train:]
		y_train = y[0:ix_train]
		y_test = y[ix_train:]

		return x_train, x_test, y_train, y_test

	def scale(self, x_train, x_test):

		self.scaler = MinMaxScaler(feature_range = [0,1])#RobustScaler(with_centering=False, with_scaling=True, quantile_range=(10.0, 90.0), copy=True)#MinMaxScaler(feature_range = [0,1])
		x_train = self.scaler.fit_transform(x_train)
		x_test = self.scaler.transform(x_test)

		return x_train, x_test


	def train(self):

		# train model
		ls_x = self.X_train
		ls_y = self.Y_train
		assert not np.any(np.isnan(ls_x) | np.isinf(ls_x))

		print self.feature_list
		feature_count = len(self.feature_list)
		x_test = self.X_test

		'''clf = Sequential()
		clf.add(Dense(128, input_shape=(feature_count,)))
		clf.add(Activation('relu'))
		clf.add(Dense(64))
		clf.add(Activation('relu'))
		clf.add(Dense(12))
		clf.add(Activation('relu'))
		clf.add(Dense(1))
		clf.add(Activation('sigmoid'))'''

		ls_x = np.reshape(ls_x, [ls_x.shape[0], 1, ls_x.shape[1]])
		x_test = np.reshape(self.X_test, [self.X_test.shape[0], 1, self.X_test.shape[1]])

		clf = Sequential()
		clf.add(LSTM(4, input_dim=feature_count))
		clf.add(Activation('relu'))
		'''clf.add(LSTM(4, return_sequences=True))
		clf.add(Activation('relu'))
		clf.add(LSTM(2))
		clf.add(Activation('relu'))'''
		clf.add(Dense(1))
		clf.add(Activation('sigmoid'))

		clf.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
		clf.fit(ls_x, ls_y, batch_size=1000, validation_data=(x_test, self.Y_test), nb_epoch=20)

		return clf

	def predict_y(self):

		assert self.model is not None
		try:
			x_test = np.reshape(self.X_test, [self.X_test.shape[0], 1, self.X_test.shape[1]])
			y_predictions = self.model.predict(x_test)
		except:
			y_predictions = self.model.predict(self.X_test)

		y_predictions = np.round(y_predictions)

		return y_predictions

	def predict_px(self):

		assert self.model is not None
		try:
			x_test = np.reshape(self.X_test, [self.X_test.shape[0], 1, self.X_test.shape[1]])
			px = self.model.predict_proba(x_test)
		except:
			px = self.model.predict_proba(self.X_test)

		return px

	def evaluate(self):
		# evaluate model
		assert self.model is not None
		self.predictions = self.predict_y()
		self.px = self.predict_px()
		print self.predictions
		score = accuracy_score(self.Y_test, self.predictions)
		return score

	def export(self):
		# export model
		with open('model.pkl', 'wb') as model:
			pickle.dump(self.model, model)
		# export scaler
		if self.scaler is not None:
			with open('scaler.pkl', 'wb') as scaler:
				pickle.dump(self.scaler, scaler)

	def save_context(self):

		if self.predictions is None:
			self.predictions = self.predict_y()
			self.px = self.predict_px()

		assert(len(self.x)==len(self.predictions))
		df_predictions = pd.DataFrame(zip(self.predictions,self.px[:,1]), index=self.x.index, columns=['y_predict','px'])
		context = pd.concat([self.x, self.y, df_predictions], axis=1)
		context['correct'] = np.zeros(context['CLOSE'].shape)
		context['correct'].loc[context['y_predict']==context['y']]=1
		context.to_csv('context.csv')

	def forwardTest(self, data):
		child = Model(data, self.options)
		price_info_cols = list(child.raw_data.columns.values)
		x = child.x.drop(price_info_cols,1).as_matrix()
		if self.options['scale'] == True:
			assert self.scaler is not None
			x = self.scaler.transform(x)
		y = child.y.values.ravel()
		try:
			px = self.model.predict_proba(x)
		except:
			x = np.reshape(x, [x.shape[0], 1, x.shape[1]])
			px = self.model.predict_proba(x)

		child.score = accuracy_score(y, np.round(px))
		print child.score
		# Result formatting for backtest
		try:
			df_result = pd.DataFrame(zip(y,px[:,1]))
		except:
			print y.shape
			print px.shape
			px = [float(x) for x in px]
			print min(px)
			print max(px)
			df_result = pd.DataFrame(zip(y,px))
		df_result.to_csv('forward_test.csv')

def main():

	options = {'time_period': 5,
				'split': 0.5,
				'classification_method': 'on_close',
				'scale': True,
				'hour_start': 9,
				'hour_end': 18}

	my_model = Model('EURUSD1_2016a.csv', options)
	forward = Model('EURUSD1_2016b.csv', options)
	
	# TRIAL

	# import and preprocess data
	scaler = MinMaxScaler(feature_range = [0,1])
	X = scaler.fit_transform(my_model.x[my_model.feature_list].as_matrix())
	y = my_model.y.as_matrix()
	y = np.reshape(y, [len(y),])

	X_forward = scaler.transform(forward.x[forward.feature_list].as_matrix())
	y_forward = forward.y.as_matrix()
	y_forward = np.reshape(y_forward, [len(y_forward),])

	n_folds = 5
	skf = StratifiedKFold(n_folds)

	classifiers = [RandomForestClassifier(n_estimators=100, n_jobs=-1, min_samples_split = 250, criterion='gini'),
					MLPClassifier(activation='relu', hidden_layer_sizes=(100,))] # SVC(kernel='linear', probability=True),GaussianNB()
	px_train = np.zeros((X.shape[0],len(classifiers)))
	px_test = np.zeros((X_forward.shape[0],len(classifiers)))

	for j, clf in enumerate(classifiers):
		print j, clf
		px_test_fold = np.zeros((X_forward.shape[0], n_folds))
		for i, (train,test) in enumerate(skf.split(X,y)):
			px_fold = clf.fit(X[train],y[train]).predict_proba(X[test])[:,1]
			px_train[test,j] = px_fold
			print px_train[test[0],:]
			# Forward test using current fold model - note: no fold applied to forward/test set
			px_test_fold[:,i] = clf.predict_proba(X_forward)[:,1]
		px_test[:,j] = px_test_fold.mean(1)

	# Decider 
	decider = LogisticRegression()
	decider.fit(px_train, y)

	px = decider.predict_proba(px_test)[:,1]
	y_predict = decider.predict(px_test)

	# Summary
	score = accuracy_score(y_forward, y_predict)
	print score

	df_result = pd.DataFrame(zip(y_forward,px))
	df_result.to_csv('forward_test.csv')

if __name__ == "__main__":

	try:
		main()
	except KeyboardInterrupt:
		print('Interupted...Exiting...')
