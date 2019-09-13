import sys
from functions import *
from agent.agent import Agent
from trading import *
from distutils.util import strtobool


if len(sys.argv) != 5:
	print('Usage: python3 train.py [stock] [window] [episodes] [is_eval]')
	exit()

ticker, window_size, episode_count, is_evaluating = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), strtobool(sys.argv[4])
init_cash = 1000000
#要給checkpoint個路徑
#c_path = "models/{}/training.ckpt".format(ticker)
m_path = "models/{}/model_weights".format(ticker)
#取得歷史資料
frequency = 'day'  # day, minute, hour
data_quantity = 400
df = get_data(ticker, data_quantity, frequency)
#起始各個class
trading = Trading(init_cash)
trading.print_log = is_evaluating
#資料整合轉換
data, time_data = init_data(df)
# n-step return
step_n = 1
#給agent初始化輸入的緯度
input_shape, neurons = get_shape(data[:window_size], window_size)
agent = Agent(ticker, input_shape, neurons, m_path, is_eval=is_evaluating)
l = len(data) - step_n

if not is_evaluating:
	target_update = 0  # 每train個幾次就update
	train_count = 0
	memory_heatup = agent.batch_size * 5  # 預先跑個幾輪亂數，再來依照權重學習
else:
	trading_record = np.zeros((1,len(data)))  # 記錄用途

# 開始
for e in range(1, episode_count + 1):
	trading.total_profit, trading.total_reward = 0, 0
	trading.win_count, trading.lose_count = 0, 0
	trading.max_con_lose = 0
	for t in range(window_size, l):         #前面的資料要來預熱一下
		state = getState(data, t, window_size)
		next_state = getState(data, t + step_n, window_size) 
		# 輸出action的位置
		action = agent.act(state)
		trading.reward = 0
		# trading policy 會直接跑完未來的一次進出，所以給整個data
		trading.policy(action, data[t:], time_data[t:])
		# 紀錄最大連續虧損
		if trading.lose_count > trading.max_con_lose:
			trading.max_con_lose = trading.lose_count
		done = True if t == l - 1 else False

		if not is_evaluating: 
			agent.append_sample(state, action, trading.reward, next_state, done)
			# 紀錄存入多少記憶	
			train_count += 1
		else:
			trading_record[0][t] = action
		
		# 更新總 reward
		trading.total_reward += trading.reward
		
		if not is_evaluating:
			# 動作一定次數才會訓練
			if train_count > 30 and train_count > memory_heatup:
				agent.train_model()
				print('Loss: %.6f' % agent.epoch_loss_avg.result().numpy())
				agent.model.save_weights(agent.checkpoint_path, save_format='tf')
				train_count = 0
				memory_heatup = 0  # 一開始heatup後就歸0，不再作用
				target_update +=1
				
			# 多次training後更新target model
			if target_update == 5 :
				agent.update_target_model()
				target_update = 0

		if done:
			if agent.epsilon > agent.epsilon_min:  #貪婪度遞減   
				agent.epsilon *= agent.epsilon_decay 
			if not trading.win_count+trading.lose_count == 0:
				traded_times = trading.win_count + trading.lose_count
				win_r = 100 * trading.win_count / traded_times
			else:
				win_r = 0
				traded_times = 0
			print("-"*124)
			print("Episode " + str(e) + "/" + str(episode_count)
			+ " | Total Profit: " + formatPrice(trading.total_profit)
			+ " | Return Ratio: %.2f%%" % round(100 * trading.total_profit / trading.init_cash, 2)
			+ " | Total Reward: " + str(round(trading.total_reward,2)))
			print("Traded : " + str(traded_times)
			+ " | Win Rate: %.2f%%" % round(win_r,2)
			+ " | Max Cont Lose: " + str(trading.max_con_lose))
			print("-"*124)
			#if is_evaluating:
			#	visualization(data, time_data, trading_record)
			