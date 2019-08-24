import numpy as np
import random, math
import pandas_datareader as pdr
import pandas as pd
from scipy import special
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler((-1,1))

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))
	
#state資料標準化
def minmaxscale(x):
	return scaler.fit_transform(x)

# returns an n-day state representation ending at time t
def getState(data, t, n):
	res = minmaxscale(data[t+1-n:t+1])
	return np.array([res])  #修正input形狀

#model的輸入值起始
def get_shape(data,window_size):
	input_shape = getState(data,window_size,window_size)
	neurons = input_shape.shape[1] * input_shape.shape[2] * 2 / 3
	return input_shape.shape[1:], math.ceil(neurons)

#計算unit要訂多少
def get_unit(avg_price,init_cash):
	unit = init_cash / avg_price
	unit = int(unit/10)
	if unit < 1:
		print('Need more cash for this target!!')
	else:
		return unit

#初始化輸入資料
def init_data(df):
	df['shift_close'] = df['Close'].shift(1)   #交易的時候，只知道昨天的收盤價
	df.dropna(how='any',inplace=True)
	data = df[['shift_close','Open','High','Low','Volume']].values
	return data

def get_long_account(inventory,close_price,commission):
	sum = 0
	for order in inventory:
		sum += order[0]
	avg_price = sum / len(inventory)
	account_profit = close_price*(1-commission) - avg_price
	return account_profit, avg_price

def get_short_account(inventory,close_price,commission):
	sum = 0
	for order in inventory:
		sum += order[0]
	avg_price = sum / len(inventory)
	account_profit = avg_price - close_price*(1+commission)
	return account_profit, avg_price

def get_inventory_value(inventory,close_price,commission):
	if inventory[0][1] == 'long':
		return close_price*(1-commission)*len(inventory)
	else:
		account_profit, avg_price = get_short_account(inventory,close_price,commission)
		return (avg_price + account_profit)*len(inventory)