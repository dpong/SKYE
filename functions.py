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
	res = minmaxscale(data[t-n:t])
	return np.array([res])  #修正input形狀

#model的輸入值起始
def get_shape(data,window_size):
	input_shape = getState(data,window_size,window_size)
	neurons = input_shape.shape[1] * input_shape.shape[2] * 2 / 3
	return input_shape.shape[1:], math.ceil(neurons)

#取得歷史資料
def get_data(ticker, start, end):
	df = pdr.DataReader('{}'.format(ticker),'yahoo', start=start, end=end)
	return df

#計算unit要訂多少
def get_unit(price, profolio):
	unit = profolio / price
	unit = int(unit/10)
	if unit < 1:
		print('Need more cash for this target!!')
	else:
		return unit

#初始化輸入資料
def init_data(df, init_cash):
	df['shift_close'] = df['Close'].shift(1)   #交易的時候，只知道昨天的收盤價
	df.dropna(how='any',inplace=True)
	df['Cash'] = init_cash
	df['Holding'] = 0
	data = df[['shift_close','Open','High','Low','Volume','Cash','Holding']].values
	return data

def get_long_account(inventory,close_price,commission):
	value_sum = [0,0,0]
	for order in inventory:
		value_sum[0] += order[0] * order[2]  #close
		value_sum[1] += order[1] * order[2]  #price
		value_sum[2] += close_price*(1-commission) * order[2]
	account_profit = value_sum[2] - value_sum[1]
	return account_profit, value_sum[1], value_sum[0]

def get_short_account(inventory,close_price,commission):
	value_sum = [0,0,0]
	for order in inventory:
		value_sum[0] += order[0] * order[2]
		value_sum[1] += order[1] * order[2]
		value_sum[2] += close_price*(1+commission) * order[2]
	account_profit = value_sum[1] - value_sum[2]
	return account_profit, value_sum[1], value_sum[0]

def get_inventory_value(inventory, close_price, commission):
	if inventory[0][-1] == 'long':
		value_sum = 0
		for order in inventory:
			value_sum += close_price * (1-commission) * order[2]
		return value_sum

	else:
		account_profit, price_value, close_value = get_short_account(inventory,close_price,commission)
		return price_value + account_profit

def get_inventory_units(inventory):
	value_sum = 0
	for order in inventory:
		value_sum += order[2]
	return value_sum