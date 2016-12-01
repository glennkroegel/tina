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
						 	'hour_start': 21,
						 	'hour_end': 23}

test = Model('EURGBP1_201415.csv', options)
test.model = joblib.load('model.pkl')

# Forward test function
price_info_cols = list(test.raw_data.columns.values)
x = test.x.drop(price_info_cols,1).as_matrix()
y = test.y.values.ravel()
px = test.model.predict_proba(x)
test.score = test.model.score(x, y)
print test.score

# Result formatting for backtest
df_result = pd.DataFrame(zip(y,px[:,1]))
df_result.to_csv('forward_test.csv')

