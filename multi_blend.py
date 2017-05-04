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
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from talib import abstract

class Preprocessor(object):
	"""docstring for ClassName"""
	def __init__(self, data, options):
		# Constructor
		self.options = options
		self.raw_data = self.getData(data)
		self.y = self.binaryClassification()
		self.x = self.features()
		self.x, self.y = self.prepareData()
		self.feature_list = [col for col in self.x if col not in self.raw_data.columns]
		
	def getData(self, data):
		# load pricing data
		self.raw_data = pd.read_csv(data)
		self.raw_data = self.raw_data.set_index("DATETIME")
		return self.raw_data

	def features(self):
		# raw feature calculation - done first before filters
		# to do - patterns and auto-correlation
		x = copy.deepcopy(self.raw_data)
		
		x['WILLR'] = taCalcIndicator(x, 'WILLR', window = 30)
		x['WILLR_D1'] = x['WILLR'].diff()
		x['WILLR_D2'] = x['WILLR'].diff(2)
		x['WILLR_D5'] = x['WILLR'].diff(5)

		#x['RWILLR'] = (x['WILLR_D1']-x['WILLR_D1'].rolling(center=False, window=30).min())/(x['WILLR_D1'].rolling(center=False,window=30).max() - x['WILLR_D1'].rolling(center=False,window=30).min())
		
		#x['ADOSC'] = taCalcIndicator(x, 'ADOSC', window = 30)
		#x['ADOSC_D1'] = x['ADOSC'].diff()

		#x['ULTOSC'] = taCalcIndicator(x, 'ULTOSC', window = 30)
		#x['ULTOSC_D1'] = x['ULTOSC'].diff()

		x['RSI'] = taCalcIndicator(x, 'RSI', window = 30)
		x['RSI_D1'] = x['RSI'].diff()
		x['RSI_D2'] = x['RSI'].diff(2)
		x['RSI_D5'] = x['RSI'].diff(5)

		#x['RRSI'] = (x['RSI_D1']-x['RSI_D1'].rolling(center=False, window=30).min())/(x['RSI_D1'].rolling(center=False,window=30).max() - x['RSI_D1'].rolling(center=False,window=30).min())

		#x['CCI'] = taCalcIndicator(x, 'CCI', window = 30)
		#x['CCI_D1'] = x['CCI'].diff()

		#x['BOP'] = taCalcIndicator(x, 'BOP')
		#x['dBOP'] = x['BOP'].diff()

		#x['ATR'] = taCalcIndicator(x, 'ATR', window = 14)
		#x['dATR'] = x['ATR'].diff()

		x['ADX'] = taCalcIndicator(x, 'ADX', window = 14)
		'''x['RADX'] = (x['ADX']-x['ADX'].rolling(center=False, window=30).min())/(x['ADX'].rolling(center=False,window=30).max() - x['ADX'].rolling(center=False,window=30).min())
		x['dADX'] = x['ADX'].diff()

		x['ROC'] = taCalcIndicator(x, 'ROC')
		x['R20'] = (x['ROC']-x['ROC'].rolling(center=False, window=30).min())/(x['ROC'].rolling(center=False,window=30).max() - x['ROC'].rolling(center=False,window=30).min())
		del x['ROC']'''

		#x['sigma'] = x['CLOSE'].rolling(window = 30, center = False).std()
		#x['dsigma'] = x['sigma'].diff()

		#x['BP5'] = breakawayEvent(x, window =5)
		x['BP10'] = breakawayEvent(x, window =10)
		x['BP15'] = breakawayEvent(x, window =15)
		x['BP30'] = breakawayEvent(x, window =30)
		'''x['BP31'] = breakawayEvent(x, window =31)
		x['BP32'] = breakawayEvent(x, window =32)
		x['BP33'] = breakawayEvent(x, window =33)
		x['BP34'] = breakawayEvent(x, window =34)
		x['BP35'] = breakawayEvent(x, window =35)
		x['BP36'] = breakawayEvent(x, window =36)
		x['BP40'] = breakawayEvent(x, window =40)'''
		x['BP60'] = breakawayEvent(x, window =60)
		#x['BP120'] = breakawayEvent(x, window =120)

		#x['H10'] = x['BP10'].rolling(window=10, center=False).sum()
		x['H20'] = x['BP30'].rolling(window=20, center=False).sum()
		#x['H30'] = x['BP10'].rolling(window=30, center=False).sum()

		# Moving averages
		#x['SMA20-40'] = x['CLOSE'].rolling(window=20, center=False).mean()-x['CLOSE'].rolling(window=100, center=False).mean()
		#x['SMA50-200'] = x['CLOSE'].rolling(window=50, center=False).mean()-x['CLOSE'].rolling(window=200, center=False).mean()
		
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
  			x.index = pd.to_datetime(x.index, format='%Y-%m-%d %H:%M:%S')'''
		
		#x = pd.concat([x, hour_dummies(x)], axis=1)
		#x['hour'] = x.index.hour/100

		#x = pd.concat([x, pattern_gen(x)], axis=1)
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
		temp = temp.loc[x['BP10'] != 0]
		#temp = temp.loc[x['ADX']<18]

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

def split(X,y,p=0.5):

	ix_train = int(p*len(X))
	X_train = X[0:ix_train]
	X_test = X[ix_train:]
	y_train = y[0:ix_train]
	y_test = y[ix_train:]

	return X_train, X_test, y_train, y_test 

def lstm_classifier(X,y):

	'''
	Keras LSTM network
	'''

	n_rows, n_cols = X.shape

	def create_model():
		# create model
		model = Sequential()
		model.add(LSTM(6, return_sequences=True, input_dim=n_cols, activation = 'relu'))
		#model.add(LSTM(32, return_sequences=True, activation = 'relu')) # , W_constraint=maxnorm(3)
		#model.add(LSTM(4, return_sequences=True, activation = 'relu'))
		model.add(LSTM(2, activation = 'relu'))
		model.add(Dense(1, activation = 'sigmoid'))
		# compile model
		#sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		return model

	# evaluate
	seed = 10
	np.random.seed(seed)
	model = KerasClassifier(build_fn=create_model, nb_epoch=30, batch_size=100) 
	
	return model

def dense_classifier(X,y):

	'''
	Keras dense network
	'''

	n_rows, n_cols = X.shape

	def create_model():
		# create model
		model = Sequential()
		model.add(Dropout(0.2, input_shape=(n_cols,)))
		model.add(Dense(40, activation = 'relu')) # , input_dim=n_cols
		model.add(Dropout(0.2))
		model.add(Dense(20, activation = 'relu', W_constraint=maxnorm(3))) # , W_constraint=maxnorm(3)
		model.add(Dropout(0.2))
		model.add(Dense(10, activation = 'relu', W_constraint=maxnorm(3)))
		model.add(Dropout(0.2))
		model.add(Dense(1, activation = 'sigmoid'))
		# compile model
		#sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		return model

	# evaluate
	seed = 30
	np.random.seed(seed)
	model = KerasClassifier(build_fn=create_model, nb_epoch=30, batch_size=100) 
	
	return model

def main():

	options = {'time_period': 5,
				'split': 0.5,
				'classification_method': 'on_close',
				'scale': True,
				'hour_start': 0,
				'hour_end': 23}

	train_data = Preprocessor('USDJPY1_201415.csv', options)
	test_data = Preprocessor('USDJPY1_201617.csv', options)

	test_data.y.to_csv('y.csv') # export test set to get datetime index in backtest. improve. 

	X = train_data.x.as_matrix()
	y = train_data.y.as_matrix()
	y = np.reshape(y, [len(y),])

	X_forward = test_data.x.as_matrix()
	y_forward = test_data.y.as_matrix()
	y_forward = np.reshape(y_forward, [len(y_forward),])

	# Scale
	scaler = MinMaxScaler(feature_range = [0,1]) #RobustScaler(with_centering=False, with_scaling=True, quantile_range=(1.0, 99.0), copy=True)#MinMaxScaler(feature_range = [0,1])
	X = scaler.fit_transform(X)
	X_forward = scaler.transform(X_forward)

	# Modeling

	# Split - Method 1
	#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, train_size = 0.8, random_state = 32)

	# Split - Method 2 (for lstm)
	X_train, X_test, y_train, y_test = split(X,y,0.5)

	'''classifiers = [RandomForestClassifier(n_estimators=1000, n_jobs=-1, min_samples_split = 250, criterion='gini'),
					dense_classifier(X_train, y_train), lstm_classifier(X_train, y_train)]
					'''

	#classifiers = [RandomForestClassifier(n_estimators = 2000, max_features = None , criterion = 'gini', min_samples_leaf = 250, n_jobs = -1,random_state = 62, class_weight = 'balanced', bootstrap = False, min_weight_fraction_leaf = 0.0),dense_classifier(X_train, y_train)]
	#classifiers = [RandomForestClassifier(n_estimators=2000, n_jobs=-1, min_samples_split = 250, criterion='gini'), dense_classifier(X_train, y_train), LogisticRegression()]
	classifiers = [dense_classifier(X_train, y_train), lstm_classifier(X_train, y_train)]

	px_train = np.zeros((X_test.shape[0],len(classifiers)))
	px_test = np.zeros((X_forward.shape[0],len(classifiers)))

	for j, clf in enumerate(classifiers):
		print clf
		try:
			clf.fit(X_train,y_train)
		except:
			X_train = np.reshape(X_train, [X_train.shape[0], 1, X_train.shape[1]])
			X_test = np.reshape(X_test, [X_test.shape[0], 1, X_test.shape[1]])
			X_forward = np.reshape(X_forward, [X_forward.shape[0], 1, X_forward.shape[1]])
			clf.fit(X_train,y_train)
		try:
			px_train[:,j] = clf.predict_proba(X_test)[:,1]
		except:
			px_train[:,j] = clf.predict(X_test)[:,1]
		px_test[:,j] = clf.predict_proba(X_forward)[:,1]

	print px_test[1000:1005,:]

	# Decider 
	#decider = CalibratedClassifierCV(LogisticRegression(C=1, class_weight='balanced', solver='lbfgs'), method='isotonic')
	decider = LogisticRegression(C=1, class_weight='balanced', solver='lbfgs')
	#decider = CalibratedClassifierCV(SVC(probability=True), method='isotonic')
	#decider = RandomForestClassifier(n_estimators=100, min_samples_split=100)
	decider.fit(px_train, y_test)

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
