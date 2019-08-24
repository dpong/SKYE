from functions import *
import sys
from agent.agent import Agent
from action import Action
import caffeine

if len(sys.argv) != 4:
	print('Usage: python3 train.py [stock] [window] [episodes]')
	exit()

caffeine.on(display=False) #電腦不休眠
ticker, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
init_cash = 1000000   
#要給checkpoint個路徑
c_path = "models/{}/training.ckpt".format(ticker)
m_path = "models/{}/model.h5".format(ticker)
#取得歷史資料
df = pdr.DataReader('{}'.format(ticker),'yahoo',start='2018-1-1',end='2019-1-1')
unit = get_unit(df['Close'].mean(),init_cash) #目前都是操作固定單位
trading = Action(unit)
#資料整合轉換
data = init_data(df)
#給agent初始化輸入的緯度
input_shape, neurons = get_shape(data[:window_size+1],window_size)
agent = Agent(ticker, input_shape, neurons, c_path)

l = len(data) -1
n_close = 0

for e in range(1, episode_count + 1):
	total_profit, cash, total_reward = 0, init_cash, 0
	inventory = []
	trading.highest_value[:] = 0
	max_drawdown = 0
	for t in range(window_size+1, l):         #前面的資料要來預熱一下
		state = getState(data, t, window_size)
		next_state = getState(data, t + 1, window_size)

		if t == l - 1: #最後一個state
			action = 3
		else:
			action = agent.act(state)
		
		trading.reward = 0

		#這邊交易的價格用當日的收盤價(t+1)代替，實際交易就是成交價格
		if action == 1 and len(inventory) > 0 and inventory[0][1]=='short':
			cash, inventory, total_profit = trading._long_clean(data[t+1][n_close] , cash, inventory, total_profit, e, episode_count,t,l)
		
		elif action == 1 and len(inventory) > 0 and inventory[0][1]=='long':
			if trading.safe_margin * cash > data[t+1][n_close] * unit:
				cash, inventory, total_profit = trading._long_new(data[t+1][n_close] , cash, inventory, total_profit, e, episode_count,t,l)
			else:
				action = 0

		elif action == 1 and len(inventory) == 0:
			if trading.safe_margin * cash > data[t+1][n_close] * unit:
				cash, inventory, total_profit = trading._long_new(data[t+1][n_close] , cash, inventory, total_profit, e, episode_count,t,l)
			else:
				action = 0

		elif action == 2 and len(inventory) > 0 and inventory[0][1]=='long':
			cash, inventory, total_profit = trading._short_clean(data[t+1][n_close] , cash, inventory, total_profit, e, episode_count,t,l)

		elif action == 2 and len(inventory) > 0 and inventory[0][1]=='short':
			if trading.safe_margin * cash > data[t+1][n_close] * unit:
				cash, inventory, total_profit = trading._short_new(data[t+1][n_close] , cash, inventory, total_profit, e, episode_count,t,l)
			else:
				action = 0
		
		elif action == 2 and len(inventory) == 0:
			if trading.safe_margin * cash > data[t+1][n_close] * unit:
				cash, inventory, total_profit = trading._short_new(data[t+1][n_close] , cash, inventory, total_profit, e, episode_count,t,l)
			else:
				action = 0

		elif action == 3 and len(inventory) > 0:
			cash, inventory, total_profit = trading._clean_inventory(data[t+1][n_close] , cash, inventory, total_profit, e, episode_count,t,l)
		
		elif action == 3 and len(inventory) == 0:
			action = 0

		if action == 0: #不動作
			trading._hold(data[t+1][n_close] , cash, inventory, e, episode_count,t,l)
				
		done = True if t == l - 1 else False
		trading.reward *= 10 #稍微放大尺度
		total_reward += trading.reward
		agent.append_sample(state, action, trading.reward, next_state, done)

		#訓練過程分5段來更新traget_model
		if t % int(l/5) == 0:
			agent.update_target_model()

		#計算max drawdown
		if len(inventory) > 0:
			inventory_value = get_inventory_value(inventory,data[t+1][n_close],trading.commission)
			inventory_value *= trading.unit
			profolio = inventory_value + cash
		else:
			profolio = cash
		if profolio - init_cash < 0:  #虧損時才做
			drawdown = (profolio - init_cash) / init_cash
			if drawdown < max_drawdown:
				max_drawdown = drawdown


		if agent.memory.tree.n_entries > agent.batch_size:
			agent.train_model()

		if done:
			print("-"*124)
			print("Episode " + str(e) + "/" + str(episode_count)
			+ " | Cash: " + formatPrice(cash) 
			+ " | Total Profit: " + formatPrice(total_profit)
			+ " | Return Ratio: %.2f%%" % round(100*total_profit/init_cash,2)
			+ " | Max DrawDown: %.2f%%" % round(-max_drawdown*100,2)
			+ " | Total Reward: " + str(round(total_reward,2)))
			print("-"*124)
			if e == episode_count:
				caffeine.off() #讓電腦回去休眠
		
		
			