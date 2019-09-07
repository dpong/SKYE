import numpy as np
import random, math
#import pandas_datareader as pdr
import pandas as pd
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()


# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns an n-day state representation ending at time t
def getState(data, t, n):
	handle = data[t-n:t,:-1]  # 避開Date
	out = scaler.fit_transform(handle)  
	out = np.array([out])  #修正input形狀
	out.dtype = 'float64'
	return out

#model的輸入值起始
def get_shape(data,window_size):
	input_shape = getState(data, window_size, window_size)
	neurons = input_shape.shape[1] * input_shape.shape[2] * 2 / 3
	return input_shape.shape[1:], math.ceil(neurons)

#取得歷史資料
def get_data(ticker, start, end):
	#df = pdr.DataReader('{}'.format(ticker),'yahoo', start=start, end=end)
	df = pd.read_csv('{}_stock_price_train.csv'.format(ticker))
	return df

#初始化輸入資料
def init_data(df, init_cash):
	df['shift_close'] = df['Close'].shift(1)   #交易的時候，只知道昨天的收盤價
	df.dropna(how='any',inplace=True)
	df['Cash'] = init_cash
	df['Holding'] = 0
	df_time = df['Date']
	data = df[['shift_close','Open','High','Low','Volume','Holding','Cash']].values
	time_data = df_time.values
	return data, time_data

def get_long_account(inventory,close_price,commission):
	value_sum = np.zeros((3))
	for order in inventory:
		value_sum[0] += order[0] * order[2]  #close
		value_sum[1] += order[1] * order[2]  #price
		value_sum[2] += close_price*(1-commission) * order[2]
	account_profit = value_sum[2] - value_sum[1]
	return account_profit, value_sum[1], value_sum[0]

def get_short_account(inventory,close_price,commission):
	value_sum = np.zeros((3))
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

# 把 inventory 換算成一個
def inventory_ensemble(inventory): 
	order_type = inventory[0][-1]
	value_sum = np.zeros((3))  # 0存close，1存price，2存unit
	for order in inventory:
		value_sum[0] += order[0] * order[2]  # close value sum
		value_sum[1] += order[1] * order[2]  # price value sum
		value_sum[2] += order[2]			 # unit sum
	avg_close = value_sum[0] / value_sum[2]  # avg close
	avg_price = value_sum[1] / value_sum[2]  # avg price
	return [[avg_close, avg_price, value_sum[2], order_type]]

def visualization(data, time_data, trading_record, return_ratio):
	import matplotlib.pyplot as plt
	from datetime import datetime
	df = pd.DataFrame(data)
	df.rename(columns={0:"Close", 1:"Open", 2:'High', 3:"Low", 4:"Volume", 5:"Cash", 6:"Holding"}, inplace=True)
	df['Date'] = pd.DataFrame(time_data)
	df['Action'] = pd.DataFrame(trading_record[0])
	df['Trade Unit'] = pd.DataFrame(trading_record[1])
	start = time_data[0]
	end = time_data[-1]
	plt.style.use('ggplot')
	plt.title("Results from " + start[:10] + " to " + end[:10]) 
	plt.xlabel("Date") 
	plt.ylabel("Close")
	plt.plot(df['Date'], df['Close'],color='blue')
	plt.scatter(df.loc[df['Action'] ==1 , 'Date'].values, df.loc[df['Action'] ==1, 'Close'].values,
		label='skitscat', color='green', s=20, marker="^")
	plt.scatter(df.loc[df['Action'] ==2 , 'Date'].values, df.loc[df['Action'] ==2, 'Close'].values,
		label='skitscat', color='red', s=20, marker="v")
	plt.scatter(df.loc[df['Action'] ==3 , 'Date'].values, df.loc[df['Action'] ==3, 'Close'].values,
		label='skitscat', color='black', s=20, marker="x")
	plt.xticks(())
	plt.show()



	

	