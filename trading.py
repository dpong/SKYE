from functions import *
from profolio import *
import numpy as np

class Trading():
    def __init__(self,init_cash):
        self.safe_margin = 0.8  #現金安全水位
        self.highest_value = np.array([0,0])  #0是多倉位，1是空倉位
        self.reward = 0
        self.reward_boost = 1  #reward值放大倍率，加速
        self.commission = 0.003  #千分之三的手續費
        self.stop_pct = 0.1  #停損%數
        self.pr_ratio = 1  #獲利/風險給reward的比例，讓agent趨向報酬會是規避風險
        self.init_cash = init_cash
        self.cash = init_cash
        self.inventory = []  #存入價格資訊：close, price, units, 動作
        self.total_profit = 0
        self.total_reward = 0
        self.profolio = Profolio(init_cash)
        self.win_count = 0
        self.lose_count = 0
        self.max_con_lose = 0
        self.print_log = False

    def policy(self, action, close, e, episode_count, t, l):
        self.profolio.total_value(close, self.cash, self.inventory, self.commission)
        unit = get_unit(close, self.profolio.profolio_value)
        if action == 1 and len(self.inventory) > 0 and self.inventory[0][-1]=='short':
            self._long_clean(close, e, episode_count, t, l)
        
        elif action == 1 and len(self.inventory) > 0 and self.inventory[0][-1]=='long':
            if self.safe_margin * self.cash > close * unit:
                self._long_new(close, e, episode_count, t, l)
            else:
                self._hold(close, e, episode_count, t, l)  # 好像不該改 action
                self.reward += -0.05
        
        elif action == 1 and len(self.inventory) == 0:
            if self.safe_margin * self.cash > close * unit:
                self._long_new_empty(close, e, episode_count, t, l)
            else:
                self._hold(close, e, episode_count, t, l)
                self.reward += -0.05
        
        elif action == 2 and len(self.inventory) > 0 and self.inventory[0][-1]=='long':
            self._short_clean(close, e, episode_count, t, l)
        
        elif action == 2 and len(self.inventory) > 0 and self.inventory[0][-1]=='short':
            if self.safe_margin * self.cash > close * unit:
                self._short_new(close, e, episode_count, t, l)
            else:
                self._hold(close, e, episode_count, t, l)
                self.reward += -0.05
        
        elif action == 2 and len(self.inventory) == 0:
            if self.safe_margin * self.cash > close * unit:
                self._short_new_empty(close, e, episode_count, t, l)
            else:
                self._hold(close, e, episode_count, t, l)
                self.reward += -0.05
        
        elif action == 3 and len(self.inventory) > 0:
            self._clean_inventory(close, e, episode_count, t, l)
        
        elif action == 3 and len(self.inventory) == 0:
            self._hold(close, e, episode_count, t, l)
            self.reward += -0.05
        
        if action == 0: #不動作
            self._hold(close, e, episode_count, t, l)

        self.reward *= self.reward_boost #放大尺度
        return action


    def _hold(self, close, e, episode_count, t, l):
        if len(self.inventory) > 0:
            if self.inventory[0][-1] == 'long':
                account_profit, price_value, close_value = get_long_account(self.inventory,close,self.commission)
                value_diff = (account_profit - self.highest_value[0]) / price_value
                if account_profit > self.highest_value[0]: 
                    self.highest_value[0] = account_profit
                #elif value_diff <= -self.stop_pct and self.highest_value[0] > 0:  #帳面獲利減少的懲罰
                #    self.reward = value_diff
                elif account_profit / price_value < -self.stop_pct:  #帳損超過的懲罰
                    self.reward = account_profit / price_value
                total_units = get_inventory_units(self.inventory)
                if self.print_log == True:
                    print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
                    + " | Bull: "+ str(total_units) + ' | Potential: ' + formatPrice(account_profit)
                    + " | Reward: " + str(round(self.reward,2)))	
                self.highest_value[1] = 0
            else:
                account_profit, price_value, close_value = get_short_account(self.inventory,close,self.commission)
                value_diff = (account_profit - self.highest_value[1]) / price_value
                if account_profit > self.highest_value[1]: 
                    self.highest_value[1] = account_profit
                #elif value_diff <= -self.stop_pct and self.highest_value[1] > 0:  #帳面獲利減少的懲罰
                #    self.reward = value_diff
                elif account_profit / price_value < -self.stop_pct:  #帳損超過的懲罰
                    self.reward = account_profit / price_value
                total_units = get_inventory_units(self.inventory)
                if self.print_log == True:
                    print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
                    + " | Bear: "+ str(total_units) + ' | Potential: ' + formatPrice(account_profit)
                    + " | Reward: " + str(round(self.reward,2)))
                self.highest_value[0] = 0
        if len(self.inventory) == 0:
            if self.print_log == True:
                print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
                + " | Nuetrual")
            self.highest_value[:] = 0
    
    #用買進動作平倉
    def _long_clean(self, close, e, episode_count, t, l):
        price = close * (1+self.commission)
        sold_price = self.inventory.pop(0)
        profit = (sold_price[1] - price) * sold_price[2]  # 2號是unit
        if profit > 0 :
            self.win_count += 1
        elif profit < 0 :
            self.lose_count += 1
        self.total_profit += profit
        self.reward = self.pr_ratio * profit / (sold_price[0] * sold_price[2])
        self.cash += profit + sold_price[0] * sold_price[2]
        total_units = get_inventory_units(self.inventory)
        if self.print_log == True:
            print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
            + " | Bear: "+ str(total_units) +" | Long: " + formatPrice(price) 
            + " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(self.total_profit)
            + " | Reward: " + str(round(self.reward,2)))
            
    def _long_new(self, close, e, episode_count, t, l):
        account_profit, price_value, close_value = get_long_account(self.inventory,close,self.commission)
        value_diff = (account_profit - self.highest_value[0]) / price_value
        if account_profit > self.highest_value[0]: 
            self.highest_value[0] = account_profit
        #elif value_diff <= -self.stop_pct and self.highest_value[0] > 0:  #帳面獲利減少的懲罰
        #    self.reward = value_diff
        elif account_profit / price_value < -self.stop_pct:  #帳損超過的懲罰
            self.reward = account_profit / price_value
        price = close * (1+self.commission)
        self.profolio.total_value(close, self.cash, self.inventory, self.commission)
        unit = get_unit(close, self.profolio.profolio_value)
        cost = close * unit
        self.cash -= cost
        self.inventory.append([close, price, unit, 'long']) #存入進場資訊
        total_units = get_inventory_units(self.inventory)
        if self.print_log == True:
            print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
            + " | Bull: "+ str(total_units) + " | Long : " + formatPrice(price)
            + ' | Potential: ' + formatPrice(account_profit)
            + " | Reward: " + str(round(self.reward,2)))

    def _long_new_empty(self, close, e, episode_count, t, l):
        price = close * (1+self.commission)
        self.profolio.total_value(close, self.cash, self.inventory, self.commission)
        unit = get_unit(close, self.profolio.profolio_value)
        cost = close * unit
        self.cash -= cost
        self.inventory.append([close, price, unit, 'long']) #存入進場資訊
        if self.print_log == True:
            print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
            + " | Bull: "+ str(unit) + " | Long : " + formatPrice(price))

    #用賣出的動作來平倉
    def _short_clean(self, close, e, episode_count, t, l):
        price = close * (1-self.commission)
        bought_price = self.inventory.pop(0)
        profit = (price - bought_price[1]) * bought_price[2]
        if profit > 0 :
            self.win_count += 1
        elif profit < 0 :
            self.lose_count += 1
        self.total_profit += profit
        self.reward = self.pr_ratio * profit / (bought_price[0] * bought_price[2])
        self.cash += profit + bought_price[0] * bought_price[2]
        total_units = get_inventory_units(self.inventory)
        if self.print_log == True:
            print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
            + " | Bull: "+ str(total_units) +" | Short: " + formatPrice(price) 
            + " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(self.total_profit)
            + " | Reward: " + str(round(self.reward,2)))

    def _short_new(self, close, e, episode_count, t, l):
        account_profit, price_value, close_value = get_short_account(self.inventory,close,self.commission)
        value_diff = (account_profit - self.highest_value[1]) / price_value
        if account_profit > self.highest_value[1]: 
            self.highest_value[1] = account_profit
        #elif value_diff <= -self.stop_pct and self.highest_value[1] > 0:  #帳面獲利減少的懲罰
        #    self.reward = value_diff
        elif account_profit / price_value < -self.stop_pct:  #帳損超過的懲罰
            self.reward = account_profit / price_value
        price = close * (1-self.commission)
        self.profolio.total_value(close, self.cash, self.inventory, self.commission)
        unit = get_unit(close, self.profolio.profolio_value)
        cost = close * unit #做空一樣要付出成本，保證金的概念
        self.cash -= cost
        self.inventory.append([close, price, unit, 'short']) #存入進場資訊
        total_units = get_inventory_units(self.inventory)
        if self.print_log == True:
            print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
            + " | Bear: "+ str(total_units) + " | Short : " + formatPrice(price)
            + ' | Potential: ' + formatPrice(account_profit)
            + " | Reward: " + str(round(self.reward,2)))
    
    def _short_new_empty(self, close, e, episode_count, t, l):
        price = close * (1 - self.commission)
        self.profolio.total_value(close, self.cash, self.inventory, self.commission)
        unit = get_unit(close, self.profolio.profolio_value)
        cost = close * unit #做空一樣要付出成本，保證金的概念
        self.cash -= cost
        self.inventory.append([close, price, unit,'short']) #存入進場資訊
        total_units = get_inventory_units(self.inventory)
        if self.print_log == True:
            print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
            + " | Bear: "+ str(total_units) + " | Short : " + formatPrice(price))
    
    #全部平倉
    def _clean_inventory(self, close, e, episode_count, t, l): 
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
                print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
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
                print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
                    + " | Clean Inventory"+ ' | Profit: ' + formatPrice(profit)
                    + " | Total Profit: " + formatPrice(self.total_profit)
                    + " | Reward: " + str(round(self.reward,2)))
        self.inventory = [] 
        self.highest_value[:] = 0

    def self_states(self, close):
        unit = get_unit(close, self.profolio.profolio_value)
        if self.safe_margin * self.cash >= close * unit:   # 判斷現金
            cash = [1,0]  # 足夠現金
        elif self.safe_margin * self.cash < close * unit:
            cash = [0,1]  # 不夠現金

        if len(self.inventory) > 0 :  # 持倉
            if self.inventory[0][-1]=='long':
                holding = [1,0]  # 多單
                account_profit, price_value, close_value = get_long_account(self.inventory,close,self.commission)
                if account_profit > 0:
                    account = [1,0,0]  # 獲利
                    if account_profit / price_value > self.stop_pct:
                        account = [1,0,1]  # 大幅獲利
                elif account_profit < 0:
                    account = [0,1,0]  # 虧損
                    if account_profit / price_value < -self.stop_pct:
                        account = [0,1,1]  # 大幅虧損
                elif account_profit == 0:
                    account = [0,0,0]  # 持平

            elif self.inventory[0][-1]=='short':
                holding = [0,1]  # 空單
                account_profit, price_value, close_value = get_short_account(self.inventory,close,self.commission)
                if account_profit > 0:
                    account = [1,0,0]  # 獲利
                    if account_profit / price_value > self.stop_pct:
                        account = [1,0,1]  # 大幅獲利
                elif account_profit < 0:
                    account = [0,1,0]  # 虧損
                    if account_profit / price_value < -self.stop_pct:
                        account = [0,1,1]  # 大幅虧損
                elif account_profit == 0:
                    account = [0,0,0]  # 持平

        else:
            holding = [0,0]  # 空手
            account = [0,0,0]  # 持平
        
        out = np.array([cash + holding + account])
        out.dtype = 'float64'
        return out
        
        
        
        
        
        
        
        
				

		