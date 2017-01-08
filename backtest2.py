#!/opt/conda/bin/python

'''
Data.py

@version: 0.1

This file is to be used in conjunction with the binary options
machine learning algorithm.

Created on April, 1, 2014

@author: Glenn Kroegel
@contact: glenn.kroegel@gmail.com
@summary: Produce performance statistics of algorithm 

'''

# Third party imports
import datetime as dt
import numpy as np
import pandas as pd
import sys
import math
import openpyxl
import xlrd
import xlwt
import urllib
import copy
from ggplot import *
from sklearn.metrics import accuracy_score

#################################################################################################################

def importData(filename):

  df = pd.read_csv(filename, index_col = 0)
  df.columns = ['Next Up', 'Probability']

  return df

def appendData(df):

  df = copy.deepcopy(df)

  df['Up Decision'] = np.zeros(df['Probability'].shape)
  df['Up Correct'] = np.zeros(df['Probability'].shape)
  df['Down Decision'] = np.zeros(df['Probability'].shape)
  df['Down Correct'] = np.zeros(df['Probability'].shape)
  df['Traded'] = np.zeros(df['Probability'].shape)
  df['Correct'] = np.zeros(df['Probability'].shape)

  return df

def getTradeAmount(balance, trade_fraction, limit = 10000):

  amount = balance*trade_fraction

  if (amount > limit):
    amount = limit
  elif (amount < 2):
    amount = 2
  else:
    pass

  return amount

#################################################################################################################


def main():
  
	# Read user input

  df_data = importData('forward_test.csv')
  df_px = df_data
  df_data = appendData(df_data)

  '''df_y = pd.read_csv('y.csv', header = None)
  df_y.columns = ['DATETIME','y']
  df_y = df_y.set_index('DATETIME')
  df_y.index = pd.to_datetime(df_y.index, format = "%Y-%m-%d %H:%M:%S")
  #print df_y.head()

  df_px.index = df_y.index'''
  df_px.to_csv('PX.csv', index_label = 'DATETIME')

  threshold_up = 0.65
  threshold_down = 0.35

  for i in range(1, len(df_data['Probability'])):

    if df_data['Probability'][i] > threshold_up:
      df_data['Up Decision'][i] = 1

    if df_data['Probability'][i] < threshold_down:
      df_data['Down Decision'][i] = 1

    if (df_data['Up Decision'][i] == 1) & (df_data['Next Up'][i] == 1):
      df_data['Up Correct'][i] = 1

    if (df_data['Down Decision'][i] == 1) & (df_data['Next Up'][i] == 0):
      df_data['Down Correct'][i] = 1

  print df_data.head(5)

  max_prob = df_data['Probability'].max()
  min_prob = df_data['Probability'].min()

  up_signals = df_data['Up Decision'].sum()
  up_proportion = df_data['Up Correct'].sum()/up_signals
  
  down_signals = df_data['Down Decision'].sum()
  down_proportion = df_data['Down Correct'].sum()/df_data['Down Decision'].sum()

  days_in_data = len(df_data['Probability'])/(11*20) # 96 fifteen minute periods in a day
  up_signals_daily = up_signals/days_in_data
  down_signals_daily = down_signals/days_in_data

  print '\nMODEL STATISTICS SUMMARY:'

  print '\nMaximum Probability: {0}\tTotal Signals: {1}\tAverage Signals Per Day: {2}'.format(max_prob, up_signals, up_signals_daily)
  print '\nMinimum Probability: {0}\tTotal Signals: {1}\tAverage Signals Per Day: {2}'.format(min_prob, down_signals, down_signals_daily)
  print '\nSuccesful Up Trades: {0}\nSuccesful Down Trades: {1}'.format(up_proportion, down_proportion)


  print '\nSIMULATION SUMMARY:\n'

  start_balance = 50
  trade_amount = 200
  win_proportion = 0.7
  trade_fraction = 0.05
  df_account_balance = pd.DataFrame(np.zeros(df_data['Up Decision'].shape), index = df_data['Up Decision'].index, columns = ['Balance']) 

  df_account_balance['Balance'][0] = start_balance

  for i in range(1, len(df_data['Up Decision'])):
    if df_data['Up Decision'][i] == 1:
      df_data['Traded'][i] = 1
      if df_data['Up Correct'][i] == 1:
        df_data['Correct'][i] = 1
        trade_amount = getTradeAmount(df_account_balance['Balance'][i-1], trade_fraction)
        #print trade_amount
        df_account_balance['Balance'][i] = df_account_balance['Balance'][i-1] + win_proportion*trade_amount
      else:
        trade_amount = getTradeAmount(df_account_balance['Balance'][i-1], trade_fraction)
        df_account_balance['Balance'][i] = df_account_balance['Balance'][i-1] - trade_amount
    if df_data['Down Decision'][i] == 1:
      df_data['Traded'][i] = 1
      if df_data['Down Correct'][i] == 1:
        df_data['Correct'][i] = 1
        trade_amount = getTradeAmount(df_account_balance['Balance'][i-1], trade_fraction)
        df_account_balance['Balance'][i] = df_account_balance['Balance'][i-1] + win_proportion*trade_amount
      else:
        trade_amount = getTradeAmount(df_account_balance['Balance'][i-1], trade_fraction)
        df_account_balance['Balance'][i] = df_account_balance['Balance'][i-1] - trade_amount
    if (df_data['Up Decision'][i] == 0) & (df_data['Down Decision'][i] == 0):
      df_account_balance['Balance'][i] = df_account_balance['Balance'].ix[i-1]

  sim_max = df_account_balance['Balance'].max()
  sim_min = df_account_balance['Balance'].min()
  sim_last = df_account_balance['Balance'][len(df_account_balance)*1/11] #28602-8640

  df_account_balance['Profit'] = df_account_balance['Balance'].diff()
  sim_std = df_account_balance['Profit'].std()
  sim_grad = df_account_balance['Profit'].mean()
  specific_variance = sim_std/sim_grad

  print 'Maximum Simulated Balance: {0}\n'.format(sim_max)
  print 'Minimum Simulated Balance: {0}\n'.format(sim_min)
  print 'Final Balance: {0}\n'.format(sim_last)
  print 'Monthly Profit: {0}\n'.format(sim_grad*1)
  print 'Specific Variance (dt = period): {0}'.format(specific_variance)

  # Draw Down
  df_trades = df_data.ix[df_data['Traded'] == 1]
  ls_locs = df_trades.index
  df_trades = df_trades.reset_index()

  draw_max = 0
  count = 0

  for i in range(0, len(df_trades)-1):
    current_val = df_trades['Correct'].ix[i]
    next_val = df_trades['Correct'].ix[i+1]
    if current_val == 0:
      count = count + 1
      if next_val == 1:
        if count > draw_max:
          draw_max = count
          max_loc = ls_locs[i]
        count = 0
    else:
      pass

  print 'Maximum Drawdown: {0}'.format(draw_max)
  print 'Location: {0}'.format(max_loc)

  # Plot Simulations
  s_x = df_account_balance.index.values
  df_account_balance['x_val'] = s_x
  p1 = ggplot(aes(x = 'x_val', y = 'Balance'), data = df_account_balance) + geom_line()
  p1 = p1 + xlab('Time') + ylab('Balance ($)') + ggtitle('')
  print p1

  s_x = df_account_balance.index.values
  s_x = df_y.index
  df_account_balance['x_val'] = s_x
  p2 = ggplot(aes(x = 'x_val', y = 'Balance'), data = df_account_balance) + geom_line() + scale_x_date()
  p2 = p2 + xlab('Time') + ylab('Balance ($)') + ggtitle('')
  print p2

  # Export
  df_account_balance.to_csv("Balance.csv")

if __name__ == "__main__":

  print("Backtesting...\n")

  try:

    main()

  except KeyboardInterrupt:

    print("Interupted...Exiting\n")









