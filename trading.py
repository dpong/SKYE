from functions import *
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.utils import to_categorical

class Trading():
    def __init__(self, init_cash):
        self.init_cash = init_cash
        self.reward = 0
        self.commission = 0.00075  # 手續費
        self.slip_cost = 10   # 滑價成本
        self.stop_pct = 0.1  # 停損%數
        self.total_profit = 0
        self.total_reward = 0
        self.win_count = 0
        self.lose_count = 0
        self.max_con_lose = 0
        self.n_close = 0
        self.print_log = False

    def policy(self, action, data, time_data):
        if action == 1:
            self._buy_hold(data, time_data)
        elif action == 2:
            self._sell_hold(data, time_data)

        elif action == 0:
            if self.print_log == True:
                print(time_data[0] + " Strategy: Do Nothing"
                + " | Total Profit: " + formatPrice(self.total_profit)
                + " | Reward: " + str(round(self.reward,2)))

    # Strategy part
    def _buy_hold(self, data, time_data):  # 做多加上移動停損
        bought_price = self._long_new(data[0, self.n_close])
        hightest_price = data[0, self.n_close]
        for t in range(1, len(data)):
            if t == len(data)-1:
                profit = self._short_clean(data[t, self.n_close], bought_price)
                break
            else:
                if data[t, self.n_close] > hightest_price:
                    hightest_price = data[t, self.n_close] 
                elif data[t, self.n_close] <= hightest_price:
                    if data[t, self.n_close] <= hightest_price * (1 - self.stop_pct):
                        profit = self._short_clean(data[t, self.n_close], bought_price)
                        break
        if profit > 0 :
            self.win_count += 1
        elif profit < 0 :
            self.lose_count += 1
        self.total_profit += profit
        self.reward = profit / bought_price
        if self.print_log == True:
            print(time_data[0] + " Strategy: %.f%% Trailing Stop Long " % (self.stop_pct*100)
            + " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(self.total_profit)
            + " | Reward: " + str(round(self.reward,2)))

    def _sell_hold(self, data, time_data):  # 做空加上移動停損
        sold_price = self._short_new(data[0, self.n_close])
        lowest_price = data[0, self.n_close]
        for t in range(1, len(data)):
            if t == len(data)-1:
                profit = self._long_clean(data[t, self.n_close], sold_price)
                break
            else:
                if data[t, self.n_close] < lowest_price:
                    lowest_price = data[t, self.n_close] 
                elif data[t, self.n_close] >= lowest_price:
                    if data[t, self.n_close] >= lowest_price * (1 + self.stop_pct):
                        profit = self._long_clean(data[t, self.n_close], sold_price)
                        break
        if profit > 0 :
            self.win_count += 1
        elif profit < 0 :
            self.lose_count += 1
        self.total_profit += profit
        self.reward = profit / sold_price
        if self.print_log == True:
            print(time_data[0] + " Strategy: %.f%% Trailing Stop Short" % (self.stop_pct*100)
            + " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(self.total_profit)
            + " | Reward: " + str(round(self.reward,2)))

    # Basic trading part
    def _long_clean(self, close, sold_price):
        price = close * (1+self.commission) + self.slip_cost
        profit = sold_price - price
        return profit
            
    def _long_new(self, close):
        price = close *(1+ self.commission) + self.slip_cost
        return price

    def _short_clean(self, close, bought_price):
        price = close *(1- self.commission) - self.slip_cost
        profit = (price - bought_price)
        return profit
    
    def _short_new(self, close):
        price = close *(1- self.commission) - self.slip_cost
        return price

    
    '''#全部平倉
    def _clean_inventory(self, close, current_time): 
        if  self.inventory[0][-1] == 'long':
            profit, price_value, close_value = get_long_account(self.inventory,close,self.commission)
            if profit > 0 :
                self.win_count += 1
            elif profit < 0 :
                self.lose_count += 1
            self.reward = self.pr_ratio * (profit / price_value)
            self.total_profit += profit
            self.cash += close_value + profit
            if self.print_log == True:
                print(current_time + " Cash: " + formatPrice(self.cash)
                    + " | Clean Inventory"+ ' | Profit: ' + formatPrice(profit)
                    + " | Total Profit: " + formatPrice(self.total_profit)
                    + " | Reward: " + str(round(self.reward,2)))
        elif self.inventory[0][-1] == 'short':
            profit, price_value, close_value = get_short_account(self.inventory,close,self.commission)
            if profit > 0 :
                self.win_count += 1
            elif profit < 0 :
                self.lose_count += 1
            self.reward = self.pr_ratio * (profit / price_value)
            self.total_profit += profit
            self.cash += close_value + profit
            if self.print_log == True:
                print(current_time + " Cash: " + formatPrice(self.cash)
                    + " | Clean Inventory"+ ' | Profit: ' + formatPrice(profit)
                    + " | Total Profit: " + formatPrice(self.total_profit)
                    + " | Reward: " + str(round(self.reward,2)))
        self.inventory = [] 
        self.highest_value[:] = 0
        '''
        
        
        
        
        
        
        
        
				

		