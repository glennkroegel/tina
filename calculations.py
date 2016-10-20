import pandas as pd
import numpy as np
import datetime as dt
import copy
from talib import abstract
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

def jarque_bera(df, application = 'WILLR_D1', window = 60):

  df = copy.deepcopy(df)

  df['jarque_bera'] = pd.rolling_apply(df[application], window, lambda x: stats.jarque_bera(x)[1])

  # bin data
  intervals = 100.0
  dx = 1.0/intervals
  df['temp'] = df['jarque_bera']

  df['temp'].loc[(df['jarque_bera'] < dx)] = 1
  df['temp'].loc[(df['jarque_bera'] >= dx)] = 2
  '''df['temp'].loc[(df['jarque_bera'] >= dx) & (df['jarque_bera'] < 2*dx)] = 2
  df['temp'].loc[(df['jarque_bera'] >= 2*dx) & (df['jarque_bera'] < 3*dx)] = 3
  df['temp'].loc[(df['jarque_bera'] >= 3*dx) & (df['jarque_bera'] < 4*dx)] = 4
  df['temp'].loc[(df['jarque_bera'] >= 4*dx)] = 5'''
  # Handle NAN issue
  return pd.get_dummies(df['temp'], prefix = 'JB')

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

  df['Normal'] = (df['CLOSE'] - pd.rolling_min(df['CLOSE'], window))/(pd.rolling_max(df['CLOSE'], window)-pd.rolling_min(df['CLOSE'], window))
  df['event'] = np.zeros(df['Normal'].shape)
  df['event'].loc[df['Normal'] == 0] = 'down'
  df['event'].loc[df['Normal'] == 1] = "up"

  return df['event']

def ribbon_sma(df):

  df = copy.deepcopy(df)

  rolling_means = {}

  for window_length in np.linspace(10,100,10):
    X = pd.rolling_mean(df['CLOSE'], window = window_length)
    rolling_means[window_length] = X
    assert(len(X) == len(df))

  rolling_means = pd.DataFrame(rolling_means, index = df.index)

  return rolling_means

def ribbon_willr(df):

  df = copy.deepcopy(df)

  series_list = {}

  for window_length in np.linspace(10,50,5):
    X = taCalcIndicator(df, 'WILLR', window = window_length)
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
  df_dummies = pd.get_dummies(df['hamming'], prefix = prefix)

  return df_dummies

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

  df['ribbon_width'] = np.round(scores,3)
  df_dummies = pd.get_dummies(df['ribbon_width'], prefix = prefix)
  return df_dummies
  #return df['ribbon_width']