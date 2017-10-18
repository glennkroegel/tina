#!/opt/conda/bin/python

import numpy as np
import pandas as pd
import talib as ta
import copy
import cPickle as pickle
from calculations import *

from sklearn import linear_model, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
import xgboost as xgb
from statsmodels.tsa.stattools import adfuller
from scipy.signal import detrend

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier

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
		self.scaler = None
		self.predictions = None
		self.px = None
		self.score = None
		
	def getData(self, data):
		# load pricing data
		self.raw_data = pd.read_csv(data)
		self.raw_data = self.raw_data.set_index("DATETIME")
		#self.raw_data = self.raw_data.iloc[0:20000]
		self.raw_data = self.raw_data
		return self.raw_data

	def features(self):
		# raw feature calculation - done first before filters
		x = copy.deepcopy(self.raw_data)

		# de-trend
		'''x['OPEN'] = x['OPEN'].rolling(center=False, window=30).apply(lambda x: detrend(x)[-1])
		x['HIGH'] = x['HIGH'].rolling(center=False, window=30).apply(lambda x: detrend(x)[-1])
		x['LOW'] = x['LOW'].rolling(center=False, window=30).apply(lambda x: detrend(x)[-1])
		x['CLOSE'] = x['CLOSE'].rolling(center=False, window=30).apply(lambda x: detrend(x)[-1])'''


		######################################################################################

		#x['f'] = x['CLOSE'].rolling(center=False, window=30).apply(lambda x: detrend(x)[-1])
		#x['df'] = x['f'].diff()

		# Dickey Fuller
		#x['p'] = x['CLOSE'].rolling(center=False, window=120).apply(lambda x: adfuller(x)[1])
		#x['dp'] = x['p'].diff()
		#x['d5'] = x['CLOSE'].pct_change(5)
		
		x['WILLR'] = taCalcIndicator(x, 'WILLR', window = 30)
		x['WILLR_D1'] = x['WILLR'].diff()
		#x['WILLR_D1'] = (x['WILLR'].diff()-x['WILLR'].diff().rolling(window=30, center=False).mean())/x['WILLR'].diff().rolling(window=30, center=False).std()
		#x['WILLR_D2'] = x['WILLR'].diff(2)
		#x['WILLR_D5'] = x['WILLR'].diff(5)

		#x['xWILLR20'] = calc_crossover(x['WILLR'],-20)
		#x['xWILLR80'] = calc_crossover(x['WILLR'],-80)
		#del x['WILLR']

		x['RSI'] = taCalcIndicator(x, 'RSI', window = 30)
		x['RSI_D1'] = x['RSI'].diff()
		#x['RSI_D1'] = (x['RSI'].diff()-x['RSI'].diff().rolling(window=30, center=False).mean())/x['RSI'].diff().rolling(window=30, center=False).std()
		#x['RSI_D2'] = x['RSI'].diff(2)
		#x['RSI_D5'] = x['RSI'].diff(5)

		#x['STD'] = x['CLOSE'].rolling(center=False, window=30).std()

		#x['xRSI30'] = calc_crossover(x['RSI'],30)
		#x['xRSI70'] = calc_crossover(x['RSI'],70)
		#del x['RSI']

		'''x['upper'], x['lower'] = taCalcBBANDS(x)
		x['X_upper'] = calc_crossover(x['CLOSE']-x['upper'],0)
		x['X_lower'] = calc_crossover(x['CLOSE']-x['lower'],0)
		del x['upper']
		del x['lower']'''

		x['CCI'] = taCalcIndicator(x, 'CCI', window = 30)
		x['CCI_D1'] = x['CCI'].diff()
		#x['CCI_D1'] = (x['CCI'].diff()-x['CCI'].diff().rolling(window=30, center=False).mean())/x['CCI'].diff().rolling(window=30, center=False).std()

		# log difference
		#x['log_diff'] = np.log(x['CLOSE'])
		#x['log_diff'] = x['log_diff'].diff()

		# MACD
		#x['macd'], x['signal'], x['histogram'] = taCalcMACD(x)
		#x['xhistogram'] = calc_crossover(x['histogram'],0)
		#del x['macd']
		#del x['signal']

		x['STOCH_K'], x['STOCH_D'] = taCalcSTOCH(x)
		#x['STOCH_X'] = calc_crossover(x['STOCH_K']-x['STOCH_D'], 0)
		#x['xSTOCH80'] = calc_crossover(x['STOCH_D'], 80)
		#x['xSTOCH20'] = calc_crossover(x['STOCH_D'], 20)
		#del x['STOCH_K']
		#del x['STOCH_D']

		#x['BOP'] = taCalcIndicator(x, 'BOP')
		#x['dBOP'] = x['BOP'].diff()

		x['ADX'] = np.log(taCalcIndicator(x, 'ADX', window = 30))
		#x['dADX'] = x['ADX'].diff()

		#x['xADX25'] = calc_crossover(x['ADX'], 25)
		#x['xADX20'] = calc_crossover(x['ADX'], 20)
		#del x['ADX']

		x['sigma'] = x['CLOSE'].rolling(window = 30, center = False).std()

		x['ATR'] = np.log(taCalcIndicator(x, 'ATR', window = 30))
	
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
		#x['H20'] = x['BP10'].rolling(window=20, center=False).sum()
		#x['H30'] = x['BP10'].rolling(window=30, center=False).sum()

		#x['dH20'] = x['H20'].diff()

		# Take break flags
		'''x['temp'] = np.absolute(x['BP30'])
		x['temp'] = x['temp'].rolling(window=3, center=False).sum()
		x['STOP1'] = np.zeros(x['temp'].shape)
		x['STOP1'].loc[x['temp']>=3] = 1 # stop condition reached
		x['STOP1'] = x['STOP1'].rolling(window=20, center=False).sum() # how often stop condition occured in period. 
		del x['temp']'''

		# Moving averages
		#x['SMA10-20'] = x['CLOSE'].rolling(window=10, center=False).mean()-x['CLOSE'].rolling(window=20, center=False).mean()
		#x['SMA50-100'] = x['CLOSE'].rolling(window=20, center=False).mean()-x['CLOSE'].rolling(window=100, center=False).mean()
		#x['SMA50-200'] = x['CLOSE'].rolling(window=50, center=False).mean()-x['CLOSE'].rolling(window=200, center=False).mean()
		
		#x['xSMA'] = calc_crossover(x['SMA10-20'],0)
		#del x['SMA10-20']

		# jarque-bera
		#x = pd.concat([x, jarque_bera(x)], axis=1)

		# breakout-points
		#x = pd.concat([x, breakout_points(x, 30, 2)], axis=1)

		# ribbon - sma
		x = pd.concat([x, width_metric(ribbon_sma(x))], axis=1)
		x = pd.concat([x, distance_metric(ribbon_sma(x))], axis=1)

		# ribbon - willr
		#x = pd.concat([x, width_metric(ribbon_willr(x), prefix='willr_hamming')], axis=1)

		# Hour dummies
		'''try:
			x.index = pd.to_datetime(x.index, format='%d/%m/%Y %H:%M')
0  		except:
  			x.index = pd.to_datetime(x.index, format='%Y-%m-%d %H:%M:%S')
		
		x = pd.concat([x, hour_dummies(x)], axis=1)'''

		# Kalman
		#x['kalman'] = WLS(x['CLOSE'])


		# Formulaic Alphas
		#x['A101'] = A101(x)
		

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
		#temp = temp.loc[x['BP30'] == 0]
		temp = temp.loc[x['BP10'] != 0]
		#temp = temp.loc[np.absolute(x['f'])>0.00025]
		#temp = temp.loc[np.absolute(x['RSI']-50)>20]
		#temp = temp.loc[x['BP30'] != 0]
		#temp = temp.loc[x['ribbon_width']<=10]
		#temp = temp.loc[x['p']<=0.5]
		#temp = temp.loc[np.absolute(x['H20'] <= 2)]
		temp = temp.loc[x['ADX']>3]
		#temp = temp.loc[x['ATR']<=-3.0]
		#temp = temp.loc[np.absolute(x['RSI_D1'])<=8]
		#temp = temp.loc[np.absolute(x['CCI'])>=100]
		#temp = temp.loc[np.absolute(x['WILLR_D1'])<=25]
		#temp = temp.loc[x['STOP1'] == 0]

		try:
			temp.index = pd.to_datetime(temp.index, format = "%d/%m/%Y %H:%M")
		except:
			temp.index = pd.to_datetime(temp.index, format = "%Y-%m-%d %H:%M:%S")

		temp = temp.between_time(dt.time(self.options['hour_start'],00), dt.time(self.options['hour_end'],00))

		print "Training size: {0}".format(len(temp))

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

		# Scale - move elsewhere
		if self.options['scale'] == True:
			self.scaler = MinMaxScaler(feature_range = [0,1])#RobustScaler(with_centering=False, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)#StandardScaler()#MinMaxScaler(feature_range = [0,1]) #RobustScaler(with_centering=False, with_scaling=True, quantile_range=(10.0, 90.0), copy=True) #MinMaxScaler(feature_range = [-1,1])
			x = self.scaler.fit_transform(x)
			#print x[1001]

		x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size = self.options['split'], random_state = 42)

		return x_train, x_test, y_train, y_test


	def train(self):
		# train model
		ls_x = self.X_train
		ls_y = self.Y_train
		# model type
		lr = LogisticRegression(C=1, class_weight='balanced', solver='lbfgs')
		lr2 = LogisticRegression(C=10, class_weight='balanced', solver='lbfgs')
		lr3 = LogisticRegression(C=1000, class_weight='balanced', solver='lbfgs')
		svm = SVC(C=10.0, probability = True)
		mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(100,))
		kn = KNeighborsClassifier(n_neighbors = 250, weights = 'distance', algorithm = 'auto', n_jobs = -1)
		rf = RandomForestClassifier(n_estimators = 1000, max_features = None , criterion = 'gini', min_samples_leaf = 250, n_jobs = -1,
									random_state = 62, class_weight = 'balanced', bootstrap = True, oob_score = True, min_weight_fraction_leaf = 0.00)
		# boosting methods
		gb = GradientBoostingClassifier(n_estimators = 1000, min_samples_leaf = 250)
		bc1 = BaggingClassifier(base_estimator = DecisionTreeClassifier(min_samples_leaf=250)
								, n_estimators = 100, max_features = 0.5, max_samples = 0.1)
		bc2 = BaggingClassifier(base_estimator = KNeighborsClassifier(n_neighbors = 250, weights = 'distance', algorithm = 'auto', n_jobs = -1)
								, n_estimators = 25, max_features = 0.5, max_samples = 0.1)
		
		ab1 = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(min_samples_leaf=250)
								, n_estimators = 1000, learning_rate = 0.5)
		ab2 = AdaBoostClassifier(base_estimator = KNeighborsClassifier(n_neighbors = 250, weights = 'distance', algorithm = 'auto', n_jobs = -1)
								, n_estimators = 10, learning_rate = 0.5)

		
		
		# classifiers
		clf1 = lr #CalibratedClassifierCV(lr, method='sigmoid', cv=3)
		clf2 = CalibratedClassifierCV(mlp, method='sigmoid', cv=3)
		clf3 = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
		clf4 = CalibratedClassifierCV(gb, method='sigmoid', cv=3)
		clf5 = lr2 #CalibratedClassifierCV(lr2, method='sigmoid', cv=3)
		clf6 = lr3 #CalibratedClassifierCV(lr3, method='sigmoid', cv=3)
		clf7 = kn
		clf12 = svm
		# boosting classifiers
		clf8 = CalibratedClassifierCV(bc1, method='sigmoid', cv=3)
		clf9 = bc2 #CalibratedClassifierCV(bc2, method='sigmoid', cv=3)
		clf10 = CalibratedClassifierCV(ab1, method='sigmoid', cv=3)
		clf11 = ab2 #CalibratedClassifierCV(ab2, method='sigmoid', cv=3)

		estimators = [('model_1',clf1), ('model_2',clf2), ('model_3',clf3), ('model_4',clf4), ('model_5',clf5), ('model_6',clf6),('model_1',clf8),('model_1',clf10)]
		#estimators = [('model_1',clf1), ('model_2',clf2), ('model_3',clf3), ('model_4',clf4)]
		#estimators = [('model_1',clf4), ('model_2',clf3)]
		#estimators = [('model_1',clf8),('model_2',clf10)]
		#estimators = [('model_1',clf8),('model_2',clf10)]
		#estimators = [('model_1',clf10)]
		#estimators = [('model_1',clf7),('model_1',clf8)]
		#estimators = [('model_1',clf1), ('model_2',clf5), ('model_3',clf6)]

		# voting
		clf = VotingClassifier(estimators=estimators, voting='soft')
		#clf = CalibratedClassifierCV(clf, method='sigmoid', cv=3)

		assert not np.any(np.isnan(ls_x) | np.isinf(ls_x))
		clf.fit(ls_x, ls_y)
		return clf

	def predict_y(self):

		assert self.model is not None
		y_predictions = self.model.predict(self.X_test)

		return y_predictions

	def predict_px(self):

		assert self.model is not None
		px = self.model.predict_proba(self.X_test)

		return px

	def evaluate(self):
		# evaluate model
		assert self.model is not None
		self.predictions = self.predict_y()
		self.px = self.predict_px()

		score = self.model.score(self.X_test, self.Y_test)
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
		px = self.model.predict_proba(x)
		child.score = self.model.score(x, y)
		print child.score

		# Result formatting for backtest
		df_result = pd.DataFrame(zip(y,px[:,1]))
		df_result.to_csv('forward_test.csv')

def dense_classifier(X,y):

	'''
	Keras dense network
	'''

	n_rows, n_cols = X.shape

	def create_model():
		# create model
		model = Sequential()
		model.add(Dropout(0.2, input_shape=(n_cols,)))
		model.add(Dense(10, activation = 'relu')) # , input_dim=n_cols
		model.add(Dropout(0.2))
		model.add(Dense(4, activation = 'relu', W_constraint=maxnorm(3))) # , W_constraint=maxnorm(3)
		model.add(Dropout(0.2))	
		model.add(Dense(1, activation = 'sigmoid'))
		# compile model
		#sgd = SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		return model

	# evaluate
	seed = 30
	np.random.seed(seed)
	model = KerasClassifier(build_fn=create_model, nb_epoch=20, batch_size=1000) 
	
	return model

def main():

	options = {'time_period': 5,
				'split': 0.7,
				'classification_method': 'on_close',
				'scale': False,
				'hour_start': 9,
				'hour_end': 18}

	my_model = Model('GBPUSD1_201415.csv', options)
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
