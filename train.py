import sys, caffeine
from functions import *
from agent.agent import Agent
from trading import Trading
from profolio import Profolio


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
#起始各個class
trading = Trading()
trading.unit = get_unit(df['Close'].mean(),init_cash) #目前都是操作固定單位
trading.init_cash = init_cash
profolio = Profolio(init_cash)
#資料整合轉換
data = init_data(df)
#給agent初始化輸入的緯度
input_shape, neurons = get_shape(data[:window_size+1],window_size)
agent = Agent(ticker, input_shape, neurons, c_path, is_eval=False)

l = len(data) -1
n_close = 0

for e in range(1, episode_count + 1):
	trading.total_profit, trading.cash, trading.total_reward = 0, init_cash, 0
	trading.inventory = []
	trading.highest_value[:] = 0
	profolio.max_drawdown = 0
	for t in range(window_size+1, l):         #前面的資料要來預熱一下
		state = getState(data, t, window_size)
		next_state = getState(data, t + 1, window_size)

		if t == l - 1: #最後一個state
			action = 3
		else:
			action = agent.act(state)
		
		trading.reward = 0
		#這邊交易的價格用當日的收盤價(t+1)代替，實際交易就是成交價格
		traded_action = trading.policy(action, data[t+1][n_close], e, episode_count, t, l)
				
		done = True if t == l - 1 else False
	
		agent.append_sample(state, traded_action, trading.reward, next_state, done)

		#訓練過程分5段來更新traget_model
		if t % int(l/5) == 0:
			agent.update_target_model()
	
		#計算max drawdown
		profolio.eval_draw_down(trading.unit, data[t+1][n_close], trading.cash, trading.inventory, trading.commission)

		if agent.memory.tree.n_entries > agent.batch_size:
			agent.train_model()

		if done:
			print("-"*124)
			print("Episode " + str(e) + "/" + str(episode_count)
			+ " | Profolio: " + formatPrice(profolio.profolio_value) 
			+ " | Total Profit: " + formatPrice(trading.total_profit)
			+ " | Return Ratio: %.2f%%" % profolio.return_ratio(trading.total_profit)
			+ " | Max DrawDown: %.2f%%" % round(-profolio.max_drawdown*100,2)
			+ " | Total Reward: " + str(round(trading.total_reward,2)))
			print("-"*124)
			if e == episode_count:
				caffeine.off() #讓電腦回去休眠
		
		
			