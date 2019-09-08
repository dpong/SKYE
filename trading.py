from functions import *
from profolio import *
from tensorflow.keras.utils import to_categorical
import numpy as np

class Trading():
    def __init__(self,init_cash):
        self.safe_margin = 0.8  #現金安全水位
        self.highest_value = np.array([0,0])  #0是多倉位，1是空倉位
        self.unit = 0
        self.reward = 0
        self.commission = 0.003  #千分之三的手續費
        self.stop_pct = 0.1  #停損%數
        self.pr_ratio = 2  #獲利/風險給reward的比例，讓agent趨向報酬會是規避風險
        self.init_cash = init_cash
        self.cash = init_cash
        self.inventory = []  #存入價格資訊：close, price, units, 動作(多，空單)
        self.total_profit = 0
        self.total_reward = 0
        self.profolio = Profolio(init_cash)
        self.win_count = 0
        self.lose_count = 0
        self.max_con_lose = 0
        self.print_log = False

    def _unit_adjust(self, value, close, unit_seed):   # seed最大是10，最小是1
        up_limit_unit = int((value / 5) / close)   # 最大出手就是20%的總資產
        if up_limit_unit < 10:
            up_limit_unit = int(value / close) 
            if up_limit_unit < 10:
                # 現金不夠分散unit
                return 1
        return int(unit_seed * up_limit_unit / 10)

    def policy(self, action, unit_seed, close, current_time):
        self.profolio.total_value(close, self.cash, self.inventory, self.commission)
        self.unit = self._unit_adjust(self.cash, close, unit_seed)
        
        if action == 1 and len(self.inventory) > 0 and self.inventory[0][-1]=='short':
            self._long_clean(close, current_time)
        
        elif action == 1 and len(self.inventory) > 0 and self.inventory[0][-1]=='long':
            if self.safe_margin * self.cash > close * self.unit:
                self._long_new(close, current_time)
            else:
                self._hold(close, current_time) 
                self.reward += -0.05
                self.unit = 0
                unit_seed = 0
        
        elif action == 1 and len(self.inventory) == 0:
            if self.safe_margin * self.cash > close * self.unit:
                self._long_new_empty(close, current_time)
            else:
                self._hold(close, current_time)
                self.reward += -0.05
                self.unit = 0
                unit_seed = 0
        
        elif action == 2 and len(self.inventory) > 0 and self.inventory[0][-1]=='long':
            self._short_clean(close, current_time)
        
        elif action == 2 and len(self.inventory) > 0 and self.inventory[0][-1]=='short':
            if self.safe_margin * self.cash > close * self.unit:
                self._short_new(close, current_time)
            else:
                self._hold(close, current_time)
                self.reward += -0.05
                self.unit = 0
                unit_seed = 0
        
        elif action == 2 and len(self.inventory) == 0:
            if self.safe_margin * self.cash > close * self.unit:
                self._short_new_empty(close, current_time)
            else:
                self._hold(close, current_time)
                self.reward += -0.05
                self.unit = 0
                unit_seed = 0
        
        elif action == 3 and len(self.inventory) > 0:
            self._clean_inventory(close, current_time)
            self.unit = 0
            unit_seed = 0
        
        elif action == 3 and len(self.inventory) == 0:
            self._hold(close, current_time)
            self.reward += -0.05
            self.unit = 0
            unit_seed = 0
        
        if action == 0: #不動作
            self._hold(close, current_time)
            self.unit = 0
            unit_seed = 0
       
        # 整合inventory，方便運算
        if len(self.inventory) > 1:
            self.inventory = inventory_ensemble(self.inventory)

        return action, self.unit, unit_seed


    def _hold(self, close, current_time):
        if len(self.inventory) > 0:
            if self.inventory[0][-1] == 'long':
                account_profit, price_value, close_value = get_long_account(self.inventory,close,self.commission)
                #value_diff = (account_profit - self.highest_value[0]) / price_value
                if account_profit > self.highest_value[0]: 
                    self.highest_value[0] = account_profit
                #elif value_diff <= -self.stop_pct and self.highest_value[0] > 0:  #帳面獲利減少的懲罰
                #    self.reward = value_diff
                elif account_profit / price_value < -self.stop_pct:  #帳損超過的懲罰
                    self.reward = account_profit / price_value
                total_units = get_inventory_units(self.inventory)
                if self.print_log == True:
                    print(current_time + " Cash: " + formatPrice(self.cash)
                    + " | Bull: "+ str(total_units) + ' | Potential: ' + formatPrice(account_profit)
                    + " | Reward: " + str(round(self.reward,2)))	
                self.highest_value[1] = 0
            else:
                account_profit, price_value, close_value = get_short_account(self.inventory,close,self.commission)
                #value_diff = (account_profit - self.highest_value[1]) / price_value
                if account_profit > self.highest_value[1]: 
                    self.highest_value[1] = account_profit
                #elif value_diff <= -self.stop_pct and self.highest_value[1] > 0:  #帳面獲利減少的懲罰
                #    self.reward = value_diff
                elif account_profit / price_value < -self.stop_pct:  #帳損超過的懲罰
                    self.reward = account_profit / price_value
                total_units = get_inventory_units(self.inventory)
                if self.print_log == True:
                    print(current_time + " Cash: " + formatPrice(self.cash)
                    + " | Bear: "+ str(total_units) + ' | Potential: ' + formatPrice(account_profit)
                    + " | Reward: " + str(round(self.reward,2)))
                self.highest_value[0] = 0
        if len(self.inventory) == 0:
            if self.print_log == True:
                print(current_time + " Cash: " + formatPrice(self.cash)
                + " | Nuetrual")
            self.highest_value[:] = 0
    
    #用買進動作平倉
    def _long_clean(self, close, current_time):
        price = close * (1+self.commission)
        if self.unit == self.inventory[0][2]:  # 平倉單位正好等於倉內單位
            sold_price = self.inventory.pop(0)
        elif self.unit > self.inventory[0][2]:  # 平倉單位大於倉內單位，扣除後，再long部位
            sold_price = self.inventory.pop(0)
            self.unit -= sold_price[2]
            self._long_new_empty(close, current_time)
        elif self.unit < self.inventory[0][2]:
            self.inventory[0][2] -= self.unit
            sold_price = self.inventory[0]
            sold_price[2] = self.unit
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
            print(current_time + " Cash: " + formatPrice(self.cash)
            + " | Bear: "+ str(total_units) +" | Long: " + formatPrice(price)
            + " | Units: " + str(self.unit) 
            + " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(self.total_profit)
            + " | Reward: " + str(round(self.reward,2)))
            
    def _long_new(self, close, current_time):
        account_profit, price_value, close_value = get_long_account(self.inventory,close,self.commission)
        #value_diff = (account_profit - self.highest_value[0]) / price_value
        if account_profit > self.highest_value[0]: 
            self.highest_value[0] = account_profit
        #elif value_diff <= -self.stop_pct and self.highest_value[0] > 0:  #帳面獲利減少的懲罰
        #    self.reward = value_diff
        elif account_profit / price_value < -self.stop_pct:  #帳損超過的懲罰
            self.reward = account_profit / price_value
        price = close * (1+self.commission)
        self.profolio.total_value(close, self.cash, self.inventory, self.commission)
        cost = close * self.unit
        self.cash -= cost
        self.inventory.append([close, price, self.unit, 'long']) #存入進場資訊
        total_units = get_inventory_units(self.inventory)
        if self.print_log == True:
            print(current_time + " Cash: " + formatPrice(self.cash)
            + " | Bull: " + str(total_units) + " | Long : " + formatPrice(price)
            + " | Units: " + str(self.unit)
            + ' | Potential: ' + formatPrice(account_profit)
            + " | Reward: " + str(round(self.reward,2)))

    def _long_new_empty(self, close, current_time):
        price = close * (1+self.commission)
        self.profolio.total_value(close, self.cash, self.inventory, self.commission)
        cost = close * self.unit
        self.cash -= cost
        self.inventory.append([close, price, self.unit, 'long']) #存入進場資訊
        if self.print_log == True:
            print(current_time + " Cash: " + formatPrice(self.cash)
            + " | Bull: "+ str(self.unit) + " | Long : " + formatPrice(price)
            + " | Units: " + str(self.unit))

    #用賣出的動作來平倉
    def _short_clean(self, close, current_time):
        price = close * (1-self.commission)
        if self.unit == self.inventory[0][2]:  # 平倉單位正好等於倉內單位
            bought_price = self.inventory.pop(0)
        elif self.unit > self.inventory[0][2]:  # 平倉單位大於倉內單位，扣除後，再short部位
            bought_price = self.inventory.pop(0)
            self.unit -= bought_price[2]
            self._short_new_empty(close, current_time)
        elif self.unit < self.inventory[0][2]:
            self.inventory[0][2] -= self.unit
            bought_price = self.inventory[0]
            bought_price[2] = self.unit
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
            print(current_time + " Cash: " + formatPrice(self.cash)
            + " | Bull: "+ str(total_units) +" | Short: " + formatPrice(price)
            + " | Units: " + str(self.unit)     
            + " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(self.total_profit)
            + " | Reward: " + str(round(self.reward,2)))

    def _short_new(self, close, current_time):
        account_profit, price_value, close_value = get_short_account(self.inventory,close,self.commission)
        #value_diff = (account_profit - self.highest_value[1]) / price_value
        if account_profit > self.highest_value[1]: 
            self.highest_value[1] = account_profit
        #elif value_diff <= -self.stop_pct and self.highest_value[1] > 0:  #帳面獲利減少的懲罰
        #    self.reward = value_diff
        elif account_profit / price_value < -self.stop_pct:  #帳損超過的懲罰
            self.reward = account_profit / price_value
        price = close * (1-self.commission)
        self.profolio.total_value(close, self.cash, self.inventory, self.commission)
        cost = close * self.unit #做空一樣要付出成本，保證金的概念
        self.cash -= cost
        self.inventory.append([close, price, self.unit, 'short']) #存入進場資訊
        total_units = get_inventory_units(self.inventory)
        if self.print_log == True:
            print(current_time + " Cash: " + formatPrice(self.cash)
            + " | Bear: "+ str(total_units) + " | Short : " + formatPrice(price)
            + " | Units: " + str(self.unit)
            + ' | Potential: ' + formatPrice(account_profit)
            + " | Reward: " + str(round(self.reward,2)))
    
    def _short_new_empty(self, close, current_time):
        price = close * (1 - self.commission)
        self.profolio.total_value(close, self.cash, self.inventory, self.commission)
        cost = close * self.unit #做空一樣要付出成本，保證金的概念
        self.cash -= cost
        self.inventory.append([close, price, self.unit,'short']) #存入進場資訊
        total_units = get_inventory_units(self.inventory)
        if self.print_log == True:
            print(current_time + " Cash: " + formatPrice(self.cash)
            + " | Bear: "+ str(total_units) + " | Short : " + formatPrice(price)
            + " | Units: " + str(self.unit))
    
    #全部平倉
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
    
    # 自身狀態的label
    def self_states(self, close):
        output = np.zeros((1,4), dtype='float64')  
        '''現金狀態，持倉狀態，帳面狀態，帳面是否超標
             0     1      2     3
        0   沒錢   空手   持平   正常 
        1   有錢   多單   賺錢   超標
        2   空著   空單   賠錢   超標2   
        '''
        if self.safe_margin * self.cash >= close * 5:   # 判斷現金，目前最大unit是5
            output[0,0] = 1  # 足夠現金
        elif self.safe_margin * self.cash < close * 5:
            output[0,0] = 0  # 不夠現金
        if len(self.inventory) > 0 :  # 持倉
            if self.inventory[0][-1]=='long':
                output[0,1] = 1  # 多單
                account_profit, price_value, close_value = get_long_account(self.inventory,close,self.commission)
                if account_profit > 0:
                    output[0,2] = 1  # 獲利
                    output[0,3] = 0  # 正常
                    if account_profit / price_value > self.stop_pct:
                        output[0,3] = 1  # 大幅獲利
                    if account_profit / price_value > self.stop_pct * 2:
                        output[0,3] = 2  # 大幅獲利2
                elif account_profit < 0:
                    output[0,2] = 2  # 虧損
                    output[0,3] = 0  # 正常
                    if account_profit / price_value < - self.stop_pct:
                        output[0,3] = 1  # 大幅虧損
                    if account_profit / price_value < - self.stop_pct * 2:
                        output[0,3] = 2  # 大幅虧損2
                elif account_profit == 0:
                    output[0,2] = 0  # 持平
                    output[0,3] = 0  # 正常
            elif self.inventory[0][-1]=='short':
                output[0,1] = 2  # 空單
                account_profit, price_value, close_value = get_short_account(self.inventory,close,self.commission)
                if account_profit > 0:
                    output[0,2] = 1  # 獲利
                    output[0,3] = 0  # 正常
                    if account_profit / price_value > self.stop_pct:
                        output[0,3] = 1  # 大幅獲利
                    if account_profit / price_value > self.stop_pct * 2:
                        output[0,3] = 2  # 大幅獲利2
                elif account_profit < 0:
                    output[0,2] = 2  # 虧損
                    output[0,3] = 0  # 正常
                    if account_profit / price_value < - self.stop_pct:
                        output[0,3] = 1  # 大幅虧損
                    if account_profit / price_value < - self.stop_pct * 2:
                        output[0,3] = 2  # 大幅虧損2
                elif account_profit == 0:
                    output[0,2] = 0  # 持平
                    output[0,3] = 0  # 正常
        else:
            output[0,1] = 0  # 空手
            output[0,2] = 0  # 持平
            output[0,3] = 0  # 正常
        
        one_hot_out = to_categorical(output, num_classes=3)
        return one_hot_out
        
        
        
        
        
        
        
        
				

		