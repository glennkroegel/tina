#!/usr/bin/env python

'''
Technical.py

@version: 1.0

Created on January, 22, 2014

@author: Glenn Kroegel
@contact: glenn.kroegel@gmail.com
@summary: Manually coded technical indicators

'''

# Third party imports
import datetime as dt
import numpy as np
import pandas as pd
import math
import openpyxl
import copy
from Functions import *
from talib import abstract
from pykalman import KalmanFilter

def calcRange(_df1, scaling = 'TRUE', window = 20):

  _df1 = copy.deepcopy(_df1)

  _df1['SIMPLE RANGE'] = _df1['HIGH'] - _df1['LOW']
  _df1['SIMPLE RANGE'].ix[_df1['SIMPLE RANGE'] == 0] = 0.00005
  
  if scaling == 'TRUE':
    
    _df1['RANGE'] = featureScale(_df1['SIMPLE RANGE'], window = window)
  
  else:

    _df1['RANGE'] = _df1['SIMPLE RANGE']

  return _df1['RANGE']

def calcKalman(df):

  df = copy.deepcopy(df)

  price = df['CLOSE'].values

  kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 1,
                  initial_state_covariance = 0.1,
                  observation_covariance=1,
                  transition_covariance=.01)

  state_means, state_covs = kf.filter(price)  
  df['KF'] = state_means[:,0] - price

  return df['KF']


def calcSMA(_df2, window, scaling = 'TRUE'):

  _df2 = copy.deepcopy(_df2)

  _df2['STANDARD SMA'] = pd.rolling_mean(_df2['OPEN'], window)

  if scaling == 'TRUE':
    _df2['SMA'] = featureScale(_df2['STANDARD SMA'], window)

  else:
    _df2['SMA'] = _df2['STANDARD SMA']

  return _df2['SMA']


def calcBollinger(_df3, window = 20, output = 'signal'):

  _df3 = copy.deepcopy(_df3)

  mid = pd.rolling_mean(_df3['CLOSE'], window = window)
  sd1 = pd.rolling_std(_df3['CLOSE'], window = window)

  # Exponential Weightings
  #mid = pd.ewma(_df3['CLOSE'], span = window)
  #sd1 = pd.ewmstd(_df3['CLOSE'], span = window)

  c = 1
  upper = mid + c*sd1
  lower = mid - c*sd1

  price = _df3['CLOSE']
  _df3['BOLLINGER'] = (price - mid)/sd1

  if output == 'signal':
    return _df3['BOLLINGER']
  if output == 'upper':
    return upper
  if output == 'lower':
    return lower

def calcBollingerVIDYA(df, window = 6):

  df = copy.deepcopy(df)

  F = float(2)/(window + 1)

  # CMO

  df['PX'] = df['CLOSE'].pct_change(1)
  df['CMO1'] = np.zeros(df['CLOSE'].shape) 
  df['CMO2'] = np.zeros(df['CLOSE'].shape)

  df['CMO1'].loc[df['PX'] > 0] = df['CLOSE'].diff()
  df['CMO2'].loc[df['PX'] < 0] = np.absolute(df['CLOSE'].diff())

  df['CMO1'] = pd.rolling_sum(df['CMO1'], window = window)
  df['CMO2'] = pd.rolling_sum(df['CMO2'], window = window)

  df['CMO'] = 100*((df['CMO1'] - df['CMO2'])/(df['CMO1'] + df['CMO2']))

  # VIDYA

  df['VIDYA'] = df['CLOSE']

  df['VIDYA'] = df['CLOSE']*F*np.absolute(df['CMO']) + (1-F*np.absolute(df['CMO']))*(df['VIDYA'].shift(1))
  #df['VIDYA'] = df['VIDYA'] + (1-F*np.absolute(df['CMO']))*df['VIDYA'].shift(1)

  # PARAMS

  # mid

  mid = df['VIDYA']

  # stdev

  df['ERROR'] = (df['CLOSE'] - df['VIDYA'])
  df['ERROR_SQUARED'] = df['ERROR']*df['ERROR']

  df['SUM_ERROR_SQUARED'] = pd.rolling_sum(df['ERROR_SQUARED'], window = window)

  sd = np.power((df['SUM_ERROR_SQUARED']/window), 0.5)

  # INDICATOR

  price = df['CLOSE']
  df['BOLLINGER'] = (price - mid)/sd

  return df['BOLLINGER']

def calcCMO(df, window = 20):

  df = copy.deepcopy(df)

  df['PX'] = df['CLOSE'].pct_change(1)
  df['CMO1'] = np.zeros(df['CLOSE'].shape) 
  df['CMO2'] = np.zeros(df['CLOSE'].shape)

  df['CMO1'].loc[df['PX'] > 0] = df['CLOSE'].diff()
  df['CMO2'].loc[df['PX'] < 0] = np.absolute(df['CLOSE'].diff())

  df['CMO1'] = pd.rolling_sum(df['CMO1'], window = window)
  df['CMO2'] = pd.rolling_sum(df['CMO2'], window = window)

  df['CMO'] = 100*((df['CMO1'] - df['CMO2'])/(df['CMO1'] + df['CMO2']))

  return df['CMO']

def calcRSI(_df4, t = 14):

  # RSI modification: scaled between 0 and 1 rather than 0 and 100 from convention
  _df4 = copy.deepcopy(_df4)
  
  _df4['CHANGE'] = _df4['CLOSE'].diff()

  _df4['GAIN'] = _df4['CHANGE']
  _df4['LOSS'] = _df4['CHANGE']

  _df4['GAIN'][_df4['GAIN'] <= 0] = 0
  _df4['LOSS'][_df4['LOSS'] > 0] = 0

  _df4['LOSS'] = -_df4['LOSS'] # Taking absolute values of losses

  # Calculate Average Gain

  df_gain = copy.deepcopy(_df4['GAIN'])
 
  df_gain.ix[0:t-1] = np.nan
  df_gain.ix[t] = _df4['GAIN'].ix[0:t-1].mean()

  for i in range(t+1, len(df_gain)):
    df_gain.ix[i] = (df_gain.ix[i-1]*(t-1) + _df4['GAIN'].ix[i])/t

  # Calculate Average Loss

  df_loss = copy.deepcopy(_df4['LOSS'])
 
  df_loss.ix[0:t-1] = np.nan
  df_loss.ix[t] = _df4['LOSS'].ix[0:t-1].mean()

  for i in range(t+1, len(df_gain)):
    df_loss.ix[i] = (df_loss.ix[i-1]*(t-1) + _df4['LOSS'].ix[i])/t
  
  # Calculate RS

  df_rs = df_gain/df_loss

  # Calculate RSI

  _df4['RSI'] = df_rs


  return _df4['RSI']

def calcMACD(_df5, t_short = 12, t_long = 26, t_signal = 9, scaling = 'TRUE'):

  _df5 = copy.deepcopy(_df5)

  df_long = pd.ewma(_df5['OPEN'], t_long)
  df_short = pd.ewma(_df5['OPEN'], t_short)

  df_signal = pd.ewma(df_short - df_long, t_signal)

  if scaling == 'TRUE':
    
    df_signal = featureScale(df_signal, window = 20)

  return df_signal

def calcStochastics(_df6, t = 14, method = 'SLOW', scaling = 'FALSE', window = 3):

  _df6 = copy.deepcopy(_df6)

  _df6['LOWEST LOW'] = pd.rolling_min(_df6['LOW'], t)
  _df6['HIGHEST HIGH'] = pd.rolling_max(_df6['HIGH'], t)

  _df6['A'] = _df6['CLOSE'] - _df6['LOWEST LOW']
  _df6['B'] = _df6['HIGHEST HIGH'] - _df6['LOWEST LOW']

  # Calculate signal lines

  if method == 'SLOW':
    _df6['D'] = 100*(pd.rolling_sum(_df6['A'], window)/pd.rolling_sum(_df6['B'], window))
    _df6['STOCHASTICS'] = pd.rolling_mean(_df6['D'], window)

  if method == 'FAST':
    _df6['K'] = 100*(_df6['A']/_df6['B']) # FAST STOCHASTIC
    _df6['STOCHASTICS'] = pd.rolling_mean(_df6['K'], window)

  # Scaling to ensure numbers not too large

  if scaling == 'TRUE':
    _df6['STOCHASTICS'] = featureScale(_df6['STOCHASTICS'], window = 96)

  return _df6['STOCHASTICS']

def patternFinder(df_data):

  df_data = copy.deepcopy(df_data)
  df_data['PATTERN'] = df_data['OPEN']

  start = 10
  current = start
  df_data['PATTERN'].ix[0:start] = np.nan

  while current < len(df_data):
    p1 = (df_data['OPEN'].ix[current - 10] - df_data['OPEN'].ix[current - 9])/df_data['OPEN'].ix[current - 10]
    p5 = (df_data['OPEN'].ix[current - 10] - df_data['OPEN'].ix[current - 5])/df_data['OPEN'].ix[current - 10]
    p9 = (df_data['OPEN'].ix[current - 10] - df_data['OPEN'].ix[current - 1])/df_data['OPEN'].ix[current - 10]
    p10 = (df_data['OPEN'].ix[current - 10] - df_data['OPEN'].ix[current])/df_data['OPEN'].ix[current - 10]

    p = [p1, p5, p9, p10]
    
    mean = np.mean(p)

    df_data['PATTERN'].ix[current] = mean

    current = current + 1

  return df_data['PATTERN']

def calcVolatility(df_6, window = 1000, scaling = 'FALSE'):

  df_6 = copy.deepcopy(df_6)

  df_6['PX-CHANGE'] = df_6['CLOSE'].pct_change()*100

  df_6['VOLATILITY'] = pd.rolling_std(df_6['PX-CHANGE'], window)

  if scaling == 'TRUE':
    df_6['VOLATILITY'] = featureScale(df_6['VOLATILITY'], window = window)

  return df_6['VOLATILITY']

def calcAroon(df_7, window = 25, output = 'oscillator'):

  df_7 = copy.deepcopy(df_7)

  maxlag = lambda x: np.argmax(x[::-1])
  minlag = lambda x: np.argmin(x[::-1])

  days_max = pd.rolling_apply(df_7['CLOSE'], func = maxlag, window = window, min_periods = 0).astype(int)
  days_min = pd.rolling_apply(df_7['CLOSE'], func = minlag, window = window, min_periods = 0).astype(int)

  df_7['UP'] = 100*(window - days_max)/window
  df_7['DOWN'] = 100*(window - days_min)/window

  df_7['AROON'] = df_7['UP'] - df_7['DOWN']

  if output == 'oscillator':
    return df_7['AROON']

  if output == 'up':
    return df_7['UP']

  if output == 'down':
    return df_7['DOWN']

def calcMAMA(df_8):

  df_8 = copy.deepcopy(df_8)

  inputs = {
      'open': df_8['OPEN'].values,
      'high': df_8['HIGH'].values,
      'low': df_8['LOW'].values,
      'close': df_8['CLOSE'].values,
      'volume': df_8['VOLUME'].values
  }

  func = abstract.Function('mama')
  values = func(inputs)
  df_8['MAMA'] = df_8['CLOSE']-values[1]

  return df_8['MAMA']

def calcStochRSI(df_9):

  df_9 = copy.deepcopy(df_9)

  inputs = {
      'open': df_9['OPEN'].values,
      'high': df_9['HIGH'].values,
      'low': df_9['LOW'].values,
      'close': df_9['CLOSE'].values,
      'volume': df_9['VOLUME'].values
  }

  func = abstract.Function('stochrsi')
  values = func(inputs)
  df_9['STOCHRSI'] = values[1]

  return df_9['STOCHRSI']


def calcSlowD(df):

  df = copy.deepcopy(df)

  inputs = {
      'open': df['OPEN'].values,
      'high': df['HIGH'].values,
      'low': df['LOW'].values,
      'close': df['CLOSE'].values,
      'volume': df['VOLUME'].values
  }

  func = abstract.Function('STOCH')
  values = func(inputs)
  df['SLOW D'] = values[1]

  return df['SLOW D']

def calcSlowK(df):

  df = copy.deepcopy(df)

  '''
  inputs = {
      'open': df['OPEN'].values,
      'high': df['HIGH'].values,
      'low': df['LOW'].values,
      'close': df['CLOSE'].values,
      'volume': df['VOLUME'].values,
      'optInFastK_Period': 3,
      'optInSlowK_Period': 14,
      'optInFastD_Period': 3,
      'optInSlowD_Period': 14,
   }
  '''

  inputs = {
      'open': df['OPEN'].values,
      'high': df['HIGH'].values,
      'low': df['LOW'].values,
      'close': df['CLOSE'].values,
      'volume': df['VOLUME'].values
   }

  func = abstract.Function('STOCH')
  values = func(inputs)
  df['SLOW K'] = values[0]

  return df['SLOW K']

def calcFullStochastic(df, period = 3):

  df = copy.deepcopy(df)

  inputs = {
      'open': df['OPEN'].values,
      'high': df['HIGH'].values,
      'low': df['LOW'].values,
      'close': df['CLOSE'].values,
      'volume': df['VOLUME'].values,
   }

  func = abstract.Function('STOCHF')
  values = func(inputs)
  df['FAST K'] = values[0]

  df['FULL K'] = pd.rolling_mean(df['FAST K'], window = period)
  df['FULL D'] = pd.rolling_mean(df['FULL K'], window = period)

  return df['FULL K'], df['FULL D']

def calcStochSignal(df):

  df = copy.deepcopy(df)

  df['STOCH K'] = calcSlowK(df)
  df['STOCH D'] = calcSlowD(df)

  df['xKDSTOCH'] = calcCrossover(df['STOCH D'], df['STOCH K'])
  df['SIGNAL'] = np.absolute(df['xKDSTOCH'])

  df['SIGNAL'].loc[(df['STOCH D'] < 80) & (df['STOCH D'] > 20)] = 0 

  return df['SIGNAL']


def calcHtFeatures(df_10):

  df_10 = copy.deepcopy(df_10)

  inputs = {
      'open': df_10['OPEN'].values,
      'high': df_10['HIGH'].values,
      'low': df_10['LOW'].values,
      'close': df_10['CLOSE'].values,
      'volume': df_10['VOLUME'].values
  }

  func = abstract.Function('HT_TRENDLINE')
  values = func(inputs)

  df_10['HT_TRENDLINE'] = values - df_10['CLOSE']

  return df_10['HT_TRENDLINE']

def calcCMF(df, window = 20):

  df = copy.deepcopy(df)

  df['MFM'] = ((df['CLOSE'] - df['LOW']) - (df['HIGH'] - df['CLOSE']))/(df['HIGH'] - df['LOW'])
  df['MFV'] = df['MFM']*df['VOLUME']

  df['CMF'] = pd.rolling_sum(df['MFV'], window = window)/pd.rolling_sum(df['VOLUME'], window = window)

  return df['CMF']

def volROC(df):

  df = copy.deepcopy(df)

  df['vROC'] = df['VOLUME'].diff()

  return df['vROC']

def calcDOSC(df):

  df['RSI'] = taCalc(df, 'RSI')
  df['SMOOTH'] = pd.ewma(pd.ewma(df['RSI'], span = 5), span = 3)
  df['SIGNAL'] = pd.rolling_mean(df['SMOOTH'], window = 9) 
  df['DOSC'] = df['SMOOTH'] - df['SIGNAL']
  #df['DOSC'] = calcCrossover(df['SMOOTH'], df['SIGNAL'])

  return df['DOSC']

def modelCorrel(df, window = 20, shift = 6):
  
  from Functions import getBinaryClassification
  
  df = copy.deepcopy(df)

  df['BBANDS'] = customCalc(df, 'BBANDS')
  df['y'] = getBinaryClassification(df)
  #df['y'].to_csv('y.csv')

  #df['rWILLR'] = pd.rolling_corr(df['WILLR'], df['y'], window = window).shift(shift)
  #df['rBOP'] = pd.rolling_corr(df['BOP'], df['y'], window = window).shift(shift)
  df['rBBANDS'] = pd.rolling_corr(df['BBANDS'], df['y'], window = window).shift(shift)

  #df['r'] = (df['rWILLR'] + df['rBOP'])/2

  df['r'] = df['rBBANDS']

  #df['r'] = df['r'].replace(np.inf,0)
  #df['r'] = df['r'].replace(np.nan,0)

  return df['r']

  #return (pd.rolling_corr(taCalc(df, 'WILLR'), getBinaryClassification(df), window = window).shift(shift) + pd.rolling_corr(taCalc(df, 'BOP'), getBinaryClassification(df), window = window).shift(shift))/2

def modelCorrelxy(df, feature, window = 20, shift = 5):
  
  from Functions import getBinaryClassification
  
  df = copy.deepcopy(df)

  df['x'] = feature
  df['y'] = getBinaryClassification(df)

  df['r'] = pd.rolling_corr(df['x'], df['y'], window = window).shift(shift)

  return df['r']

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

def taCalc_CCI(df, window = 60):

  df = copy.deepcopy(df)

  indicator = 'CCI'

  inputs = {
      'open': df['OPEN'].values.astype(float),
      'high': df['HIGH'].values.astype(float),
      'low': df['LOW'].values.astype(float),
      'close': df['CLOSE'].values.astype(float),
      'volume': df['VOLUME'].values.astype(float),
  }

  func = abstract.Function(indicator)
  values = func(inputs, window)
  df[indicator] = values

  return df[indicator]

def taCalc_WILLR(df, window = 60):

  df = copy.deepcopy(df)

  indicator = 'WILLR'

  inputs = {
      'open': df['OPEN'].values.astype(float),
      'high': df['HIGH'].values.astype(float),
      'low': df['LOW'].values.astype(float),
      'close': df['CLOSE'].values.astype(float),
      'volume': df['VOLUME'].values.astype(float),
  }

  func = abstract.Function(indicator)
  values = func(inputs, window)
  df[indicator] = values

  return df[indicator]

def calcCrossover(df_1, df_2, method = 'signal'):

  df_1 = copy.deepcopy(df_1)
  df_2 = copy.deepcopy(df_2)

  df = pd.DataFrame([df_1, df_2]).T
  df.columns = ['A','B']

  df['Crossover'] = np.where(df['A'] >= df['B'], 1, 0)
  df['Signal'] = df['Crossover'].diff()

  if method == 'signal':
    return df['Signal']
  else:
    return df['Crossover']

def dirChange(df):

  df = copy.deepcopy(df)

  df['PX-CHANGE'] = df['CLOSE'].pct_change(periods = 1)

  df['DIRECTION'] = np.zeros(df['CLOSE'].shape)
  df['DIRECTION'].loc[df['PX-CHANGE'] < 0] = -1
  df['DIRECTION'].loc[df['PX-CHANGE'] > 0] = 1

  df['temp'] = df['DIRECTION'].diff()

  df['dirChange'] = np.zeros(df['CLOSE'].shape)
  df['dirChange'].loc[df['temp'] != 0] = 1

  return df['dirChange']

def lastTick(df):

  df = copy.deepcopy(df)

  df['PX-CHANGE'] = df['CLOSE'].pct_change(periods = 1)

  df['DIRECTION'] = np.zeros(df['CLOSE'].shape)
  df['DIRECTION'].loc[df['PX-CHANGE'] < 0] = -1
  df['DIRECTION'].loc[df['PX-CHANGE'] > 0] = 1

  return df['DIRECTION']

def isDoji(df):

  df = copy.deepcopy(df)

  df['DELTA'] = df['CLOSE']-df['OPEN']

  df['DOJI'] = np.zeros(df['CLOSE'].shape)
  df['DOJI'][df['DELTA'] == 0] = 1

  return df['DOJI']

def boundaryEvent(df, threshold = 20*10**-5):

  df = copy.deepcopy(df)

  df['diff'] = df['CLOSE'].diff()
  df['event'] = np.zeros(df['diff'].shape)
  df['event'][np.absolute(df['diff']) > threshold] = 1

  return df['event']

def runawayEvent(df, threshold = 10*10**-5, window = 10):

  df = copy.deepcopy(df)

  df['diff'] = df['CLOSE'].diff(window)
  df['event'] = np.zeros(df['diff'].shape)
  df['event'][np.absolute(df['diff']) >= threshold] = 1

  return df['event']

def breakawayEvent(df, window = 30):

  df = copy.deepcopy(df)

  df['Normal'] = (df['CLOSE'] - pd.rolling_min(df['CLOSE'], window))/(pd.rolling_max(df['CLOSE'], window)-pd.rolling_min(df['CLOSE'], window))
  df['event'] = np.zeros(df['Normal'].shape)
  df['event'].loc[df['Normal'] == 0] = 1
  df['event'].loc[df['Normal'] == 1] = 1

  return df['event']

def breakawayEvent2(df, window = 30):

  df = copy.deepcopy(df)

  df['Normal'] = (df['CLOSE'] - pd.rolling_min(df['CLOSE'], window))/(pd.rolling_max(df['CLOSE'], window)-pd.rolling_min(df['CLOSE'], window))
  df['event'] = np.zeros(df['Normal'].shape)
  df['event'].loc[df['Normal'] == 0] = -1
  df['event'].loc[df['Normal'] == 1] = 1

  return df['event']

def breakawayEventDown(df, window = 30):

  df = copy.deepcopy(df)

  df['Normal'] = (df['CLOSE'] - pd.rolling_min(df['CLOSE'], window))/(pd.rolling_max(df['CLOSE'], window)-pd.rolling_min(df['CLOSE'], window))
  df['event'] = np.zeros(df['Normal'].shape)
  df['event'].loc[df['Normal'] == 0] = 1

  return df['event']

def breakawayEventUp(df, window = 30):

  df = copy.deepcopy(df)

  df['Normal'] = (df['CLOSE'] - pd.rolling_min(df['CLOSE'], window))/(pd.rolling_max(df['CLOSE'], window)-pd.rolling_min(df['CLOSE'], window))
  df['event'] = np.zeros(df['Normal'].shape)
  df['event'].loc[df['Normal'] == 1] = 1

  return df['event']

def volumeLull(df):

  df = copy.deepcopy(df)

  df['a'] = df['CLOSE'].pct_change(periods = 5)
  df['b'] = df['VOLUME'].pct_change(periods = 5)

  df['b_vol'] = np.zeros(df['CLOSE'].shape)

  df['b_vol'][(df['a'] < 0) & (df['b'] > 0)] = 1
  df['b_vol'][(df['a'] > 0) & (df['b'] < 0)] = -1

def stillMoving(df):

  df = copy.deepcopy(df)

  df['isMoving'] = np.zeros(df['CLOSE'].shape)
  df['isMoving'][df['CLOSE'] == df['HIGH']] = 1
  df['isMoving'][df['CLOSE'] == df['LOW']] = 1

  return df['isMoving']

def volSpike(df, threshold = 40):

  df = copy.deepcopy(df)

  df['volSpike'] = np.zeros(df['CLOSE'].shape)
  df['volSpike'][df['VOLUME'] >= threshold] = 1

  return df['volSpike']

def linearSlope(df, window):

  df = copy.deepcopy(df)
  df2 = df.reset_index()

  df['y'] = df['CLOSE']
  df['x'] = df2.index


  df['COV'] = pd.rolling_cov(df['x'], df['y'], window = window)
  df['VAR'] = pd.rolling_var(df['x'], window = window)

  df['SLOPE'] = df['COV']/df['VAR']

  return df['SLOPE']

def volumeSlope(df, window):

  df = copy.deepcopy(df)
  df2 = df.reset_index()

  df['y'] = df['VOLUME']
  df['x'] = df2.index


  df['COV'] = pd.rolling_cov(df['x'], df['y'], window = window)
  df['VAR'] = pd.rolling_var(df['x'], window = window)

  df['SLOPE'] = df['COV']/df['VAR']

  return df['SLOPE']

def timePeriod(df):

  df = copy.deepcopy(df)

  return 0

def customCalc(df_prices, indicator):

  df = copy.deepcopy(df_prices)

  if indicator == 'BBANDS':
    df['BBANDS'] = calcBollinger(df, window = 20)
    return df['BBANDS']

  if indicator == 'BBANDS20':
    df['BBANDS'] = calcBollinger(df, window = 20)
    return df['BBANDS']

  if indicator == 'BBANDS60':
    df['BBANDS'] = calcBollinger(df, window = 60)
    return df['BBANDS']

  if indicator == 'BBANDS5':
    df['BBANDS'] = calcBollinger(df, window = 5)
    return df['BBANDS']

  if indicator == 'RANGE':
    df['RANGE'] = calcRange(df, window = 50, scaling = 'FALSE')
    return df['RANGE']

  if indicator == 'AROON':
    df['AROON'] = calcAroon(df, window = 20)
    return df['AROON']

  if indicator == 'vROC':
    df['vROC'] = df['VOLUME'].diff()
    return df['vROC']

  if indicator == 'MAMA':
    df['MAMA'] = calcMAMA(df)
    return df['MAMA']

  if indicator == 'DOSC':
    df['DOSC'] = calcDOSC(df)
    return df['DOSC']

  if indicator == 'CMF':
    df['CMF'] = calcCMF(df, window = 20)
    return df['CMF']

  if indicator == 'HT_TRENDLINE':
    df['HT_TRENDLINE'] = calcHtFeatures(df)
    return df['HT_TRENDLINE']

  if indicator == 'STOCHRSI':
    df['STOCHRSI'] = calcStochRSI(df)
    return df['STOCHRSI']
  
  if indicator == 'MACD':
    df['MACD'] = calcMACD(df, scaling = 'FALSE')
    return df['MACD']

  if indicator == 'STOCH':
    df['STOCH'] = calcStochastics(df, method = 'SLOW', scaling = 'FALSE')
    return df['STOCH']

  if indicator == 'STOCHF':
    df['STOCHF'] = calcStochastics(df, method = 'FAST', scaling = 'FALSE')
    return df['STOCHF']
  
  if indicator == 'SMA':
    df['SMA'] = calcSMA(df, window = 50, scaling = 'FALSE')
    return df['SMA']

  if indicator == 'MA':
    df['MA'] = calcSMA(df, window = 20, scaling = 'FALSE')
    return df['MA']

  if indicator == 'dBOP':
    df['dBOP'] = taCalc(df, 'BOP')
    df['dBOP'] = df['dBOP'].diff()
    return df['dBOP']

  if indicator == 'lastTick':
    df['lastTick'] = lastTick(df)
    return df['lastTick']
  
  if indicator == '_x1':
    df['_x1'] = taCalc(df, 'BOP') #df['RSI'].diff()*np.absolute(df['BBANDS'])
    df['_x1'] = df['_x1']/df['VOLUME']
    return df['_x1']

  if indicator == '_x2':
    df['_x2'] = (df['CLOSE']-df['OPEN'])/df['VOLUME']
    return df['_x2']

  if indicator == '_x3':
    df['_x3'] = (df['CLOSE']-df['HIGH'])
    return df['_x3']

  if indicator == '_x4':
    df['_x4'] = df['CLOSE']-df['LOW']
    return df['_x4']

  if indicator == '_x5':
    df['_x5'] = (df['CLOSE']-df['OPEN'])/(df['HIGH'] - df['LOW'])
    #df['_x5'] = df['_x5'].diff()
    return df['_x5']

  if indicator == '_x6':
    df['_x6'] = df['CLOSE']-df['OPEN']/(df['VOLUME']*calcVolatility(df))
    return df['_x6']
  
  if indicator == '_x7':
    df['_x7'] = pd.rolling_kurt(df['CLOSE'], window = 20)
    return df['_x7']

  if indicator == '_x8':
    df['_x8'] = pd.rolling_skew(df['CLOSE'], window = 20)
    return df['_x8']
  
  if indicator == '_x9':
    df['_x9'] = df['CLOSE']-pd.ewma(df['CLOSE'], span = 6)
    return df['_x9']

  if indicator == '_x10':
    df['_x10'] = df['CLOSE']-pd.ewma(df['CLOSE'], span = 20)
    return df['_x10']

  if indicator == '_x11':
    df['_x11'] = df['CLOSE']-pd.ewma(df['CLOSE'], span = 16)
    return df['_x11']

  if indicator == '_x12':
    df['_x12'] = pd.ewma(df['CLOSE'], span = 6) - pd.ewma(df['CLOSE'], span = 20)
    return df['_x12']

  if indicator == '_x13':
    df['_x13'] = df['CMO']/pd.ewma(df['CMO'], span = 16)
    return df['_x13']

  if indicator == '_x14':
    df['_x14'] = df['ULTOSC']/pd.ewma(df['ULTOSC'], span = 16)
    return df['_x14']

  if indicator == '_x15':
    df['_x15'] = df['RSI']/pd.ewma(df['RSI'], span = 16)
    return df['_x15']

  if indicator == '_x16':
    df['_x16'] = df['BOP']-pd.ewma(df['BOP'], span = 16)
    return df['_x16']

  if indicator == '_x17':
    df['_x17'] = df['WILLR']-pd.ewma(df['WILLR'], span = 16)
    return df['_x17']

  if indicator == '_x18':
    df['_x18'] = df['AROON']/pd.ewma(df['AROON'], span = 16)
    return df['_x18']

  if indicator == '_x19':
    df['_x19'] = df['MINUS_DI'] - df['PLUS_DI']
    return df['_x19']

  if indicator == '_x20':
    df['_x20'] = df['STOCH'] - df['STOCHF']
    return df['_x20']

  if indicator == '_x21':
    df['_x21'] = df['OPEN'] - df['HIGH']
    return df['_x21']

  if indicator == '_x22':
    df['_x22'] = pd.rolling_mean(df['CLOSE'], window = 20) - pd.ewma(df['CLOSE'], span = 10)
    return df['_x22']

  if indicator == '_x23':
    df['a'] = df['CLOSE']
    df['b'] = df['CLOSE']
    df['_x23'] = pd.ewma(df['a'], span = 6) - pd.ewma(df['b'], span = 3)
    return df['_x23']

  if indicator == '_x24':
    df['a'] = df['CLOSE']
    df['b'] = df['CLOSE']
    df['_x24'] = pd.ewma(df['a'], span = 10) - pd.ewma(df['b'], span = 20)
    return df['_x24']

  if indicator == '_x25':
    df['a'] = df['CLOSE']
    df['b'] = df['CLOSE']
    df['_x25'] = pd.ewma(df['a'], span = 20) - pd.ewma(df['b'], span = 50)
    return df['_x25']

  if indicator == '_x26':
    df['a'] = df['CLOSE'] - df['OPEN']
    df['b'] = df['CLOSE'] - df['OPEN']
    df['_x26'] = pd.ewma(df['a'], span = 20) - pd.ewma(df['b'], span = 3)
    return df['_x26']

  if indicator == '_x27':
    df['a'] = df['CLOSE'] - df['HIGH']
    df['b'] = df['CLOSE'] - df['HIGH']
    df['_x27'] = pd.ewma(df['a'], span = 6) - pd.ewma(df['b'], span = 16)
    return df['_x27']

  if indicator == '_x28':
    df['a'] = df['CLOSE'].pct_change(periods = 1)
    df['b'] = df['CLOSE'].pct_change(periods = 10)
    df['c'] = df['CLOSE'].pct_change(periods = 50)
    df['_x28'] = (df['b'])
    return df['_x28']

  if indicator == '_x29':
    df['_x29'] = (df['CLOSE'] - df['OPEN'])/df['VOLUME']
    return df['_x29']

  if indicator == '_x30':
    df['a'] = taCalc(df,'WILLR')
    df['a'] = np.absolute(df['a'])
    df['a'][df['a'] < 1] = 1
    df['_x30'] = np.log10(df['a'].values)
    return df['_x30']

  if indicator == '_x31':
    df['a'] = taCalc(df,'WILLR')
    df['b'] = df['VOLUME']
    df['_x31'] = df['a']/df['b']
    return df['_x31']

  #########################################################

  if indicator == '_x50':
    period = 20
    df['a'] = (df['CLOSE'] - pd.rolling_mean(df['CLOSE'], window = period))/pd.rolling_std(df['CLOSE'], window = period)
    df['b'] = (df['VOLUME'] - pd.rolling_mean(df['VOLUME'], window = period))/pd.rolling_std(df['VOLUME'], window = period)
    df['_x50'] = df['a'] - df['b']
    return df['_x50']

  if indicator == '_x51':

    str_indicator = 'MFI'
    df[str_indicator] = taCalc(df, str_indicator)

    period = 10
    df['a'] = (df['CLOSE'] - pd.rolling_mean(df['CLOSE'], window = period))/pd.rolling_std(df['CLOSE'], window = period)
    df['b'] = (df[str_indicator] - pd.rolling_mean(df[str_indicator], window = period))/pd.rolling_std(df[str_indicator], window = period)
    df['_x51'] = df['a'] - df['b']
    return df['_x51']

  if indicator == '_x52':
    
    str_indicator = 'OBV'
    df[str_indicator] = taCalc(df, str_indicator)

    period = 20
    df['a'] = (df['CLOSE'] - pd.rolling_mean(df['CLOSE'], window = period))/pd.rolling_std(df['CLOSE'], window = period)
    df['b'] = (df[str_indicator] - pd.rolling_mean(df[str_indicator], window = period))/pd.rolling_std(df[str_indicator], window = period)
    df['_x52'] = df['a'] - df['b']
    return df['_x52']

  if indicator == '_x53':
    period = 20
    df['a'] = (df['CLOSE']-df['OPEN'])/(df['HIGH'] - df['LOW'])
    df['b'] = df['VOLUME'].diff()
    df['_x53'] = df['a']*df['b']
    return df['_x53']

#######################################################

  if indicator == 'b_1':
    df['a'] = pd.ewma(df['CLOSE'], span = 3)
    df['b'] = pd.ewma(df['CLOSE'], span = 6)
    df['b_1'] = calcCrossover(df['a'], df['b'])
    return df['b_1']

  if indicator == 'b_2':
    df['a'] = pd.ewma(df['CLOSE'], span = 5)
    df['b'] = pd.ewma(df['CLOSE'], span = 10)
    df['b_2'] = calcCrossover(df['a'], df['b'])
    return df['b_2']

  if indicator == 'b_3':
    df['a'] = pd.ewma(df['CLOSE'], span = 6)
    df['b'] = pd.ewma(df['CLOSE'], span = 16)
    df['b_3'] = calcCrossover(df['a'], df['b'])
    return df['b_3']

  if indicator == 'b_4':
    df['a'] = pd.ewma(df['CLOSE'], span = 10)
    df['b'] = pd.ewma(df['CLOSE'], span = 20)
    df['b_4'] = calcCrossover(df['a'], df['b'])
    return df['b_4']

  if indicator == 'b_5':
    df['a'] = pd.ewma(df['CLOSE'], span = 6)
    df['b'] = pd.ewma(df['CLOSE'], span = 20)
    df['b_5'] = calcCrossover(df['a'], df['b'])
    return df['b_5']

  if indicator == 'b_6':
    df['a'] = pd.ewma(df['CLOSE'], span = 6)
    df['b'] = pd.ewma(df['CLOSE'], span = 50)
    df['b_6'] = calcCrossover(df['a'], df['b'])
    return df['b_6']

  if indicator == 'b_7':
    df['a'] = pd.ewma(df['CLOSE'], span = 10)
    df['b'] = pd.ewma(df['CLOSE'], span = 50)
    df['b_7'] = calcCrossover(df['a'], df['b'])
    return df['b_7']

  if indicator == 'b_8':
    df['a'] = pd.ewma(df['CLOSE'], span = 20)
    df['b'] = pd.ewma(df['CLOSE'], span = 50)
    df['b_8'] = calcCrossover(df['a'], df['b'])
    return df['b_8']

  if indicator == 'b_9':
    df['a'] = pd.ewma(df['CLOSE'], span = 10)
    df['b'] = pd.ewma(df['CLOSE'], span = 100)
    df['b_9'] = calcCrossover(df['a'], df['b'])
    return df['b_9']

  if indicator == 'b_10':
    df['a'] = pd.rolling_mean(df['CLOSE'], 3)
    df['b'] = pd.rolling_mean(df['CLOSE'], 6)
    df['b_10'] = calcCrossover(df['a'], df['b'])
    return df['b_10']

  if indicator == 'b_11':
    df['a'] = pd.rolling_mean(df['CLOSE'], 5)
    df['b'] = pd.rolling_mean(df['CLOSE'], 10)
    df['b_11'] = calcCrossover(df['a'], df['b'])
    return df['b_11']

  if indicator == 'b_12':
    df['a'] = pd.rolling_mean(df['CLOSE'], 6)
    df['b'] = pd.rolling_mean(df['CLOSE'], 16)
    df['b_12'] = calcCrossover(df['a'], df['b'])
    return df['b_12']

  if indicator == 'b_13':
    df['a'] = pd.rolling_mean(df['CLOSE'], 10)
    df['b'] = pd.rolling_mean(df['CLOSE'], 20)
    df['b_13'] = calcCrossover(df['a'], df['b'])
    return df['b_13']

  if indicator == 'b_14':
    df['a'] = pd.rolling_mean(df['CLOSE'], 10)
    df['b'] = pd.rolling_mean(df['CLOSE'], 50)
    df['b_14'] = calcCrossover(df['a'], df['b'])
    return df['b_14']

  if indicator == 'b_15':
    df['a'] = pd.rolling_mean(df['CLOSE'], 20)
    df['b'] = pd.rolling_mean(df['CLOSE'], 50)
    df['b_15'] = calcCrossover(df['a'], df['b'])
    return df['b_15']

  if indicator == 'b_16':
    df['a'] = pd.ewma(df['CLOSE'], 3)
    df['b'] = pd.rolling_mean(df['CLOSE'], 6)
    df['b_16'] = calcCrossover(df['a'], df['b'])
    return df['b_16']

  if indicator == 'b_17':
    df['a'] = pd.ewma(df['CLOSE'], 16)
    df['b'] = pd.rolling_mean(df['CLOSE'], 6)
    df['b_17'] = calcCrossover(df['a'], df['b'])
    return df['b_17']

  if indicator == 'b_18':
    df['a'] = pd.ewma(df['CLOSE'], 10)
    df['b'] = pd.rolling_mean(df['CLOSE'], 10)
    df['b_18'] = calcCrossover(df['a'], df['b'])
    return df['b_18']

  if indicator == 'b_19':
    df['a'] = pd.ewma(df['CLOSE'], 20)
    df['b'] = pd.rolling_mean(df['CLOSE'], 10)
    df['b_19'] = calcCrossover(df['a'], df['b'])
    return df['b_19']

  if indicator == 'b_20':
    df['a'] = df['CLOSE']
    df['b'] = pd.ewma(df['CLOSE'], 6)
    df['b_20'] = calcCrossover(df['a'], df['b'])
    return df['b_20']

  if indicator == 'b_21':
    df['a'] = df['CLOSE']
    df['b'] = pd.ewma(df['CLOSE'], 10)
    df['b_21'] = calcCrossover(df['a'], df['b'])
    return df['b_21']

  if indicator == 'b_22':
    df['a'] = df['CLOSE']
    df['b'] = pd.ewma(df['CLOSE'], 16)
    df['b_22'] = calcCrossover(df['a'], df['b'])
    return df['b_22']

  if indicator == 'b_23':
    df['a'] = df['CLOSE']
    df['b'] = pd.ewma(df['CLOSE'], 20)
    df['b_23'] = calcCrossover(df['a'], df['b'])
    return df['b_23']

  if indicator == 'b_24':
    df['a'] = df['CLOSE']
    df['b'] = pd.ewma(df['CLOSE'], 50)
    df['b_24'] = calcCrossover(df['a'], df['b'])
    return df['b_24']

  if indicator == 'b_30':
    df['a'] = calcDOSC(df)
    df['b'] = df['CLOSE']*0
    df['b_30'] = calcCrossover(df['a'], df['b'])
    return df['b_30']

  if indicator == 'b_BOP':
    df['a'] = taCalc(df, 'BOP')
    df['b'] = df['CLOSE']*0
    df['b_30'] = calcCrossover(df['a'], df['b'])
    return df['b_30']

  if indicator == 'b_100':
    df['a'] = calcBollinger(df, window = 5, output = 'upper')
    df['b'] = df['CLOSE']
    df['b_100'] = calcCrossover(df['a'], df['b'])
    return df['b_100']

  if indicator == 'b_101':
    df['a'] = calcBollinger(df, window = 5, output = 'lower')
    df['b'] = df['CLOSE']
    df['b_101'] = calcCrossover(df['a'], df['b'])
    return df['b_101']
  
  if indicator == 'b_102':
    df['a'] = calcBollinger(df, window = 10, output = 'upper')
    df['b'] = df['CLOSE']
    df['b_102'] = calcCrossover(df['a'], df['b'], method = 'Crossover')
    return df['b_102']

  if indicator == 'b_103':
    df['a'] = calcBollinger(df, window = 10, output = 'lower')
    df['b'] = df['CLOSE']
    df['b_103'] = calcCrossover(df['a'], df['b'], method = 'Crossover')
    return df['b_103']

  if indicator == 'b_104':
    df['a'] = calcStochastics(df)
    df['b'] = df['CLOSE']*0 + 30
    df['b_104'] = calcCrossover(df['a'], df['b'])
    return df['b_104']

  if indicator == 'b_105':
    df['a'] = calcStochastics(df)
    df['b'] = df['CLOSE']*0 + 70
    df['b_105'] = calcCrossover(df['a'], df['b'])
    return df['b_105']

  if indicator == 'b_106':
    df['a'] = calcBollinger(df,window = 20)
    df['b'] = np.ones(df['CLOSE'].shape)
    df['b_106'] = calcCrossover(df['a'], df['b'])
    return df['b_106']

  if indicator == 'b_107':
    df['a'] = calcBollinger(df, window = 20)
    df['b'] = np.zeros(df['CLOSE'].shape)-1
    df['b_107'] = calcCrossover(df['a'], df['b'])
    return df['b_107']

  if indicator == 'b_vol':
    df['a'] = df['CLOSE'].pct_change(periods = 7)
    df['b'] = df['VOLUME'].pct_change(periods = 7)

    df['b_vol'] = np.zeros(df['CLOSE'].shape)

    df['b_vol'][(df['a'] < 0) & (df['b'] > 0)] = 1
    df['b_vol'][(df['a'] > 0) & (df['b'] < 0)] = -1

    return df['b_vol']

  if indicator == 'b_boundary':
    df['b_boundary'] = boundaryEvent(df, threshold = 20*10**-5)
    return df['b_boundary']

  if indicator == 'b_breakdown':
    df['breaks'] = breakawayEventDown(df, window = 60)
    df['sum'] = pd.rolling_sum(df['breaks'], window = 5)
    df['b_breakdown'] = np.zeros(df['CLOSE'].shape)
    df['b_breakdown'][(df['sum'] >= 1) & (df['sum'].diff() > 0)] = 1
    #df['b_breakdown'][(df['sum'] == 2)] = 1
    return df['b_breakdown']  

  if indicator == 'b_breakup':
    df['breaks'] = breakawayEventUp(df, window = 60)
    df['sum'] = pd.rolling_sum(df['breaks'], window = 5)
    df['b_breakup'] = np.zeros(df['CLOSE'].shape)
    df['b_breakup'][(df['sum'] >= 1) & (df['sum'].diff() > 0)] = 1
    #df['b_breakup'][(df['sum'] == 2)] = 1
    return df['b_breakup'] 





