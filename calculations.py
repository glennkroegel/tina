#!/usr/bin/env python

import pandas as pd
import numpy as np
import datetime as dt
import copy
import talib
from talib import abstract
from talib import MA_Type
from pykalman import KalmanFilter
from scipy import stats

from statsmodels import regression
import statsmodels.api as sm
import scipy.stats as stats
import scipy.spatial.distance as distance

def taCalc(df, indicator):

  df = copy.deepcopy(df)

  inputs = {
      'open': df['OPEN'].values.astype(float),
      'high': df['HIGH'].values.astype(float),
      'low': df['LOW'].values.astype(float),
      'close': df['CLOSE'].values.astype(float),
      'volume': df['VOLUME'].values.astype(float)
  }

  func = abstract.Function(indicator)
  values = func(inputs)
  df[indicator] = values

  return df[indicator]

def taCalcIndicator(df, indicator, window = 20):

  df = copy.deepcopy(df)

  inputs = {
      'open': df['OPEN'].values.astype(float),
      'high': df['HIGH'].values.astype(float),
      'low': df['LOW'].values.astype(float),
      'close': df['CLOSE'].values.astype(float),
      'volume': df['VOLUME'].values.astype(float)
  }

  func = abstract.Function(indicator)
  values = func(inputs, window)
  df[indicator] = values

  return df[indicator]

def taCalcSTOCH(df):

  df = copy.deepcopy(df)

  inputs = {
      'open': df['OPEN'].values.astype(float),
      'high': df['HIGH'].values.astype(float),
      'low': df['LOW'].values.astype(float),
      'close': df['CLOSE'].values.astype(float),
      'volume': df['VOLUME'].values.astype(float)
  }

  slowk, slowd = abstract.STOCH(inputs, 5, 3, 0, 3, 0)
  
  df['K'] = slowk
  df['D'] = slowd

  return df['K'], df['D']

def taCalcBBANDS(df):

  df = copy.deepcopy(df)

  close = df['CLOSE'].values.astype(float)
  upper, middle, lower = talib.BBANDS(close, matype = MA_Type.T3)

  #df['value'] = (middle-lower)/(upper-lower)

  df['upper'] = upper
  df['lower'] = lower

  return df['upper'], df['lower']

def taCalcMACD(df):

  close = copy.deepcopy(df['CLOSE'].values.astype(float))
  macd_talib, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
  return macd_talib, signal, hist

def pattern_gen(df, lookback = 25, prefix = 'p_'):

  df_pattern = pd.DataFrame()
  df_pattern['source'] = df['CLOSE']
  df_pattern['feature'] = df_pattern['source'].diff()

  df_pattern['value'] = np.zeros(df_pattern['feature'].shape)
  df_pattern['value'].loc[df_pattern['feature']>0] = 1
  df_pattern['value'].loc[df_pattern['feature']<=0] = 0

  df_pattern[prefix+'0'] = df_pattern['value']
  for i in range(1,lookback):
    df_pattern[prefix+str(i)] = df_pattern['value'].shift(i)
  
  del df_pattern['source']
  del df_pattern['feature']
  del df_pattern['value']

  print df_pattern.tail(10)

  return df_pattern


def jarque_bera(df, application = 'CLOSE', window = 60):

  df = copy.deepcopy(df)

  df['jarque_bera'] = pd.rolling_apply(df[application], window, lambda x: stats.jarque_bera(x)[1])

  # bin data
  '''
  intervals = 100.0
  dx = 1.0/intervals
  df['temp'] = df['jarque_bera']

  df['temp'].loc[(df['jarque_bera'] < dx)] = 1
  df['temp'].loc[(df['jarque_bera'] >= dx)] = 2
  df['temp'].loc[(df['jarque_bera'] >= dx) & (df['jarque_bera'] < 2*dx)] = 2
  df['temp'].loc[(df['jarque_bera'] >= 2*dx) & (df['jarque_bera'] < 3*dx)] = 3
  df['temp'].loc[(df['jarque_bera'] >= 3*dx) & (df['jarque_bera'] < 4*dx)] = 4
  df['temp'].loc[(df['jarque_bera'] >= 4*dx)] = 5'''
  
  # Handle NAN issue
  #return pd.get_dummies(df['temp'], prefix = 'JB')

  return df['jarque_bera']

def breakout_points(df, delta = 30, quantity = 3):

  df = copy.deepcopy(df)
  recv_cols = list(df.columns)

  # Calculate flags
  for i in range(1,quantity):
    T = i*delta
    label = 'BP'+str(T)
    df[label] = breakawayEvent(df, window=T)

  # Determine dummies
  df_dummies = pd.get_dummies(df[[col for col in df.columns if col not in recv_cols]])

  return df_dummies

def breakawayEvent(df, window = 30):

  df = copy.deepcopy(df)

  df['Normal'] = (df['CLOSE'] - df['CLOSE'].rolling(window=window, center=False).min())/(df['CLOSE'].rolling(window=window, center=False).max()-df['CLOSE'].rolling(window=window, center=False).min())
  df['event'] = np.zeros(df['Normal'].shape)
  df['event'].loc[df['Normal'] == 0] = -1 #'down'
  df['event'].loc[df['Normal'] == 1] = 1 #"up"

  return df['event']

def ribbon_sma(df):

  df = copy.deepcopy(df)

  rolling_means = {}

  for window_length in np.linspace(20,60,5):
    X = df['CLOSE'].rolling(window=int(window_length), center=False).mean()
    rolling_means[window_length] = X
    assert(len(X) == len(df))

  rolling_means = pd.DataFrame(rolling_means, index = df.index)

  return rolling_means

def ribbon_willr(df):

  df = copy.deepcopy(df)

  series_list = {}

  for window_length in np.linspace(20,60,5):
    X = taCalcIndicator(df, 'WILLR', window = window_length)
    series_list[window_length] = X
    assert(len(X) == len(df))

  series_list = pd.DataFrame(series_list, index = df.index)

  return series_list

def ribbon_rsi(df):

  df = copy.deepcopy(df)

  series_list = {}

  for window_length in np.linspace(10,190,5):
    X = taCalcIndicator(df, 'RSI', window = window_length)
    series_list[window_length] = X
    assert(len(X) == len(df))

  series_list = pd.DataFrame(series_list, index = df.index)

  return series_list

def distance_metric(df, prefix = 'hamming'):

  # Returns dummies describing hamming distance of ribbon input

  # Read in ribbon
  print df.tail()
  df = copy.deepcopy(df)

  # Rank ribbon (cols)
  scores = pd.Series(index = df.index)
  for timestamp in df.index:
    values = df.loc[timestamp]
    ranking = stats.rankdata(values)
    d = distance.hamming(ranking, range(1,len(ranking)+1))
    scores[timestamp] = d

  df['hamming'] = scores
  #df_dummies = pd.get_dummies(df['hamming'], prefix = prefix)
  #return df_dummies
  return df['hamming']

def width_metric(df, prefix = 'width'):

  # Returns dummies describing width of ribbon input

  # Read in ribbon
  print df.tail()
  df = copy.deepcopy(df)

  # Rank ribbon (cols)
  scores = pd.Series(index = df.index)
  for timestamp in df.index:
    values = df.loc[timestamp]
    d = np.max(values)-np.min(values)
    scores[timestamp] = d

  df['ribbon_width'] = scores #np.round(scores,3)
  #df_dummies = pd.get_dummies(df['ribbon_width'], prefix = prefix)
  #return df_dummies
  return df['ribbon_width']

def hour_dummies(df, prefix='hour_'):

  df = copy.deepcopy(df)

  try:
    df.index = pd.to_datetime(df.index, format='%d/%m/%Y %H:%M')
  except:
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
  
  df['hour'] = df.index.hour
  df_dummies = pd.get_dummies(df['hour'], prefix = prefix)

  return df_dummies

def calc_crossover(series, value):

  df_crossover = series*0

  df_crossover.loc[series > value] = 1
  df_crossover.loc[series < value] = 0

  df_crossover = df_crossover.diff()

  return df_crossover

def WLS(series):

  series = copy.deepcopy(series)
  close = series.as_matrix()

  obs_mat = np.vstack([close, np.ones(close.shape)]).T[:,np.newaxis]

  delta = 1e-5
  trans_cov = delta/(delta - np.eye(2))

  kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                  initial_state_mean=np.zeros(2),
                  initial_state_covariance=np.ones((2, 2)),
                  transition_matrices=np.eye(2),
                  observation_matrices=obs_mat,
                  observation_covariance=1,
                  transition_covariance=trans_cov)

  state_means, state_covs = kf.filter(close)

  slope = state_means[:,0]
  intercept = state_means[:,1]
  estimate = slope*close + intercept

  df_kalman = pd.DataFrame(index=series.index)

  # Output feature - error
  error = close - estimate
  df_kalman['error'] = error

  return df_kalman['error']

def A101(data):

  data = copy.deepcopy(data)
  
  data['A101'] = (data['CLOSE']-data['OPEN'])/((data['HIGH']-data['LOW'])+0.001)

  return data['A101']














