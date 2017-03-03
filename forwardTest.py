#!/opt/conda/bin/python

import numpy as np
import pandas as pd
import talib as ta
import copy
import cPickle as pickle
from calculations import *
from sklearn.externals import joblib

from model import Model

options = {'time_period': 5,
				'split': 0.7,
				'classification_method': 'on_close',
				'scale': True,
				'hour_start': 9,
				'hour_end': 18}

test = Model('EURUSD1_201617.csv', options)
test.model = joblib.load('model.pkl')
test.scaler = joblib.load('scaler.pkl')

# Forward test function
price_info_cols = list(test.raw_data.columns.values)
x = test.x.drop(price_info_cols,1).as_matrix()
if test.options['scale'] == True:
	assert test.scaler is not None
	x = test.scaler.transform(x)
	#print x[1000:1003]
y = test.y.values.ravel()
y_predict = test.model.predict(x)
px = test.model.predict_proba(x)
test.score = test.model.score(x, y)
print test.score

test.predictions = y_predict
test.px = px
test.save_context()

# Result formatting for backtest
df_result = pd.DataFrame(zip(y,px[:,1]))
df_result.to_csv('forward_test.csv')

