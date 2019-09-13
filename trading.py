from functions import *
from profolio import *
from tensorflow.keras.utils import to_categorical
import numpy as np

class Trading():
    def __init__(self,init_cash):
        self.safe_margin = 0.8  #現金安全水位
        self.highest_value = np.array([0,0])  #0是多倉位，1是空倉位
        self.unit = 10
        self.reward = 0
        self.commission = 0.003  #千分之三的手續費
        self.stop_pct = 0.1  #停損%數
        self.init_cash = init_cash
        self.cash = init_cash
        #self.inventory = []  #存入價格資訊：close, price, units, 動作(多，空單)
        self.total_profit = 0
        self.total_reward = 0
        self.profolio = Profolio(init_cash)
        self.win_count = 0
        self.lose_count = 0
        self.max_con_lose = 0

        self.n_close = 0
        
        self.print_log = False

    def policy(self, action, data, time_data):
        if action == 1:
            self._buy_hold(1, data, time_data)
        elif action == 2:
            self._sell_hold(1, data, time_data)
        
        elif action == 3:
            self._buy_hold(5, data, time_data)
        elif action == 4:
            self._sell_hold(5, data, time_data)

        elif action == 5:
            self._buy_hold(20, data, time_data)
        elif action == 6:
            self._sell_hold(20, data, time_data)

        elif action == 0:
            if self.print_log == True:
                print(time_data[0] + " Strategy: Do Nothing"
                + " | Total Profit: " + formatPrice(self.total_profit)
                + " | Reward: " + str(round(self.reward,2)))

    # strategy part
    def _buy_hold(self, days, data, time_data):
        bought_price = self._long_new(data[0, self.n_close])
        if len(data) >= days: 
            period = data[0:days]
            profit = self._short_clean(period[-1, self.n_close], bought_price)
        else:
            profit = self._short_clean(data[-1, self.n_close], bought_price)
        if profit > 0 :
            self.win_count += 1
        elif profit < 0 :
            self.lose_count += 1
        self.total_profit += profit
        self.reward = profit / bought_price
        if self.print_log == True:
            print(time_data[0] + " Strategy: Buy hold {} days ".format(days)
            + " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(self.total_profit)
            + " | Reward: " + str(round(self.reward,2)))

    def _sell_hold(self, days, data, time_data):
        sold_price = self._short_new(data[0, self.n_close])
        if len(data) >= days:
            period = data[0:days]
            profit = self._long_clean(period[-1, self.n_close], sold_price)
        else:
            profit = self._long_clean(data[-1, self.n_close], sold_price)
        if profit > 0 :
            self.win_count += 1
        elif profit < 0 :
            self.lose_count += 1
        self.total_profit += profit
        self.reward = profit / sold_price
        if self.print_log == True:
            print(time_data[0] + " Strategy: Sell hold {} days".format(days)
            + " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(self.total_profit)
            + " | Reward: " + str(round(self.reward,2)))

    # Basic trading part
    def _long_clean(self, close, sold_price):
        price = close * (1+self.commission)
        profit = (sold_price - price)
        return profit
            
    def _long_new(self, close):
        price = close * (1 + self.commission)
        return price

    def _short_clean(self, close, bought_price):
        price = close * (1 - self.commission)
        profit = (price - bought_price)
        return profit
    
    def _short_new(self, close):
        price = close * (1 - self.commission)
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
        
        
        
        
        
        
        
        
				

		