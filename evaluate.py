from functions import *
from agent.agent import Agent
from trading import Trading
from profolio import Profolio
import sys


ticker, window_size = 'TSLA', 20
episode_count = 1
init_cash = 1000000
#要給checkpoint個路徑
m_path = "models/{}/model_weights".format(ticker)
#取得歷史資料
start = '2018-1-1'
end = '2019-1-1'
df = get_data(ticker, start, end)
#df_ben = get_data('SPY', start, end)
#起始各個class
trading = Trading(init_cash)
trading.print_log = True
profolio = Profolio(init_cash)
#資料整合轉換
data = init_data(df, init_cash)
# n-step return
step_n = 3
#給agent初始化輸入的緯度
input_shape, neurons = get_shape(data[:window_size], window_size)
agent = Agent(ticker, input_shape, neurons, m_path, is_eval=True)


l = len(data) - step_n
n_close = 0
n_cash = -2  #cash資料放data的倒數第二個
n_holding = -1  #holding資料放data的倒數第一個

episode_count = 1

for e in range(1, episode_count + 1):
	trading.total_profit, trading.cash, trading.total_reward = 0, init_cash, 0
	trading.inventory = []
	trading.highest_value[:] = 0
	trading.win_count, trading.lose_count = 0, 0
	trading.max_con_lose = 0
	profolio.max_drawdown = 0
	data[:,n_cash] = init_cash
	data[:,n_holding] = 0
	for t in range(window_size+1, l):         #前面的資料要來預熱一下
		state = getState(data, t, window_size)
		next_state = getState(data, t + step_n, window_size) 
		self_state = trading.self_states(data[t+1, n_close])

		action = agent.act(state, self_state)

		trading.reward = 0
		#這邊交易的價格用當日的收盤價(t+1)代替，實際交易就是成交價格
		traded_action = trading.policy(action, data[t+1, n_close], e, episode_count, t, l)

		# 紀錄最大連續虧損
		if trading.lose_count > trading.max_con_lose:
			trading.max_con_lose = trading.lose_count

		done = True if t == l - 1 else False

		#計算max drawdown
		profolio.eval_draw_down(data[t+1, n_close], trading.cash, trading.inventory, trading.commission)


		#本次動作回饋到下一個的data裡
		data[t+1,n_cash] = trading.cash
		if len(trading.inventory) > 0:
			data[t+1,n_holding] = get_inventory_value(trading.inventory, data[t+1, n_close], trading.commission)
		else:
			data[t+1,n_holding] = 0 

		if done:
			if not trading.win_count+trading.lose_count ==0:
				win_r = 100 * trading.win_count / (trading.win_count+trading.lose_count)
			else:
				win_r = 0
			sharp = profolio.sharp_ratio(data, step_n)
			print("-"*124)
			print("Episode " + str(e) + "/" + str(episode_count)
			+ " | Profolio: " + formatPrice(profolio.profolio_value) 
			+ " | Total Profit: " + formatPrice(profolio.profolio_value - trading.init_cash)
			+ " | Return Ratio: %.2f%%" % round(100 * (profolio.profolio_value - trading.init_cash) / trading.init_cash,2)
			+ " | Realized Return Ratio: %.2f%%" % round(100 * trading.total_profit / trading.init_cash, 2))
			print("Max DrawDown: %.2f%%" % round(-profolio.max_drawdown*100,2)
			+ " | Sharp Ratio: %.2f%%" % sharp
			+ " | Win Rate: %.2f%%" % round(win_r,2)
			+ " | Max Cont Lose: " + str(trading.max_con_lose)
			+ " | Total Reward: " + str(round(trading.total_reward,2)))
			print("-"*124)
