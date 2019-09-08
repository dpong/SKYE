import sys
from functions import *
from agent.agent import Agent
from trading import *
from profolio import *
from distutils.util import strtobool


if len(sys.argv) != 5:
	print('Usage: python3 train.py [stock] [window] [episodes] [is_eval]')
	exit()

ticker, window_size, episode_count, is_evaluating = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), strtobool(sys.argv[4])
init_cash = 10000000
#要給checkpoint個路徑
#c_path = "models/{}/training.ckpt".format(ticker)
m_path = "models/{}/model_weights".format(ticker)
#取得歷史資料
start = '2018-1-1'
end = '2019-1-1'
df = get_data(ticker, start, end)
#起始各個class
trading = Trading(init_cash)
trading.print_log = is_evaluating
profolio = Profolio(init_cash)
#資料整合轉換
data, time_data = init_data(df, init_cash)
# n-step return
step_n = 1
#給agent初始化輸入的緯度
input_shape, neurons = get_shape(data[:window_size], window_size)
self_state = trading.self_states(data[0,0])
agent = Agent(ticker, input_shape, self_state.shape, neurons, m_path, is_eval=is_evaluating)
l = len(data) - step_n
n_close = 0
n_cash = -1  #cash資料放data的倒數第一個
n_holding = -2  #holding資料放data的倒數第二個
n_return_ratio = -3  #return ratio資料放data的倒數第三個

if not is_evaluating:
	target_update = 0  # 每train個幾次就update
	train_count = 0
	memory_heatup = agent.batch_size * 5  # 預先跑個幾輪亂數，再來依照權重學習
else:
	trading_record = np.zeros((2,len(data)))  #視覺化用途

# 開始
for e in range(1, episode_count + 1):
	trading.total_profit, trading.cash, trading.total_reward = 0, init_cash, 0
	trading.inventory = []
	trading.highest_value[:] = 0
	trading.win_count, trading.lose_count = 0, 0
	trading.max_con_lose = 0
	profolio.max_drawdown = 0
	data[:,n_cash] = init_cash
	data[:,n_holding] = 0
	for t in range(window_size, l):         #前面的資料要來預熱一下
		state = getState(data, t, window_size)
		next_state = getState(data, t + step_n, window_size) 
		self_state = trading.self_states(data[t, n_close])
		# 輸出action和unit位置
		action, unit_seed = agent.act(state, self_state)
		trading.reward = 0
		# 這邊交易的價格用當日的收盤價代替，實際交易就是成交價格
		traded_action, traded_unit, unit_seed = trading.policy(action, unit_seed, data[t, n_close], time_data[t])
		# 紀錄最大連續虧損
		if trading.lose_count > trading.max_con_lose:
			trading.max_con_lose = trading.lose_count
		done = True if t == l - 1 else False
		# 計算的時候給大reward
		if trading.total_profit / trading.init_cash <= 0.05 and done:
			trading.reward += -1
		elif trading.total_profit / trading.init_cash > 0.05 and done:
			trading.reward += 1

		if not is_evaluating:   # unit存入seed而不是調整後的
			agent.append_sample([state, self_state], [traded_action, unit_seed], trading.reward, [next_state, self_state], done)
			# 紀錄存入多少記憶	
			train_count += 1
		else:
			trading_record[0][t] = traded_action
			trading_record[1][t] = traded_unit
		
		# 更新總 reward
		trading.total_reward += trading.reward
		
		if not is_evaluating:
			# 動作一定次數才會訓練
			if train_count > agent.batch_size and train_count > memory_heatup:
				agent.train_model()
				agent.model.save_weights(agent.checkpoint_path, save_format='tf')
				train_count = 0
				memory_heatup = 0  # 一開始heatup後就歸0，不再作用
				target_update +=1
				
			# 多次training後更新target model
			if target_update == 2 :
				agent.update_target_model()
				target_update = 0

		#計算max drawdown
		profolio.eval_draw_down(data[t, n_close], trading.cash, trading.inventory, trading.commission)
		
		# 本次動作回饋到下一個的data裡，紀錄起來
		data[t+1,n_cash] = trading.cash
		data[t+1,n_return_ratio] = trading.total_profit / trading.init_cash
		if len(trading.inventory) > 0:
			data[t+1,n_holding] = get_inventory_value(trading.inventory, data[t, n_close], trading.commission)
		else:
			data[t+1,n_holding] = 0 

		if done:
			if not trading.win_count+trading.lose_count == 0:
				traded_times = trading.win_count + trading.lose_count
				win_r = 100 * trading.win_count / traded_times
			else:
				win_r = 0
				traded_times = 0
			sharp = profolio.sharp_ratio(data, step_n)
			print("-"*124)
			print("Episode " + str(e) + "/" + str(episode_count)
			+ " | Profolio: " + formatPrice(profolio.profolio_value) 
			+ " | Total Profit: " + formatPrice(profolio.profolio_value - trading.init_cash)
			+ " | Return Ratio: %.2f%%" % round(100 * (profolio.profolio_value - trading.init_cash) / trading.init_cash,2)
			+ " | Realized Return Ratio: %.2f%%" % round(100 * trading.total_profit / trading.init_cash, 2))
			print("Max DrawDown: %.2f%%" % round(-profolio.max_drawdown*100,2)
			+ " | Sharp Ratio: %.2f%%" % sharp
			+ " | Traded : " + str(traded_times)
			+ " | Win Rate: %.2f%%" % round(win_r,2)
			+ " | Max Cont Lose: " + str(trading.max_con_lose)
			+ " | Total Reward: " + str(round(trading.total_reward,2)))
			print("-"*124)
			if is_evaluating:
				visualization(data, time_data, trading_record)
			