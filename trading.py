from functions import *

class Trading():
    def __init__(self):
        self.safe_margin = 0.8  #現金安全水位
        self.unit = 0
        self.highest_value = np.array([0,0])  #0是多倉位，1是空倉位
        self.reward = 0
        self.reward_boost = 10  #reward值放大10倍，加速
        self.commission = 0.003  #千分之三的手續費
        self.stop_pct = 0.1  #停損%數
        self.pr_ratio = 1.5  #獲利/風險給reward的比例，讓agent趨向報酬會是規避風險
        self.init_cash = 0
        self.cash = self.init_cash
        self.inventory = []  #存入價格資訊：close, price, 動作
        self.total_profit = 0
        self.total_reward = 0

    def policy(self, action, close, e, episode_count, t, l):
        if action == 1 and len(self.inventory) > 0 and self.inventory[0][-1]=='short':
            self._long_clean(close, e, episode_count, t, l)
        
        elif action == 1 and len(self.inventory) > 0 and self.inventory[0][-1]=='long':
            if self.safe_margin * self.cash > close * self.unit:
                self._long_new(close, e, episode_count, t, l)
            else:
                action = 0
        
        elif action == 1 and len(self.inventory) == 0:
            if self.safe_margin * self.cash > close * self.unit:
                self._long_new_empty(close, e, episode_count, t, l)
            else:
                action = 0
        
        elif action == 2 and len(self.inventory) > 0 and self.inventory[0][-1]=='long':
            self._short_clean(close, e, episode_count, t, l)
        
        elif action == 2 and len(self.inventory) > 0 and self.inventory[0][-1]=='short':
            if self.safe_margin * self.cash > close * self.unit:
                self._short_new(close, e, episode_count, t, l)
            else:
                action = 0
        
        elif action == 2 and len(self.inventory) == 0:
            if self.safe_margin * self.cash > close * self.unit:
                self._short_new_empty(close, e, episode_count, t, l)
            else:
                action = 0
        
        elif action == 3 and len(self.inventory) > 0:
            self._clean_inventory(close, e, episode_count, t, l)
        
        elif action == 3 and len(self.inventory) == 0:
            action = 0
        
        if action == 0: #不動作
            self._hold(close, e, episode_count, t, l)

        self.reward *= self.reward_boost #放大尺度
        self.total_reward += self.reward
        return action


    def _hold(self, close, e, episode_count, t, l):
        if len(self.inventory) > 0:
            if self.inventory[0][-1] == 'long':
                account_profit, avg_price, avg_close = get_long_account(self.inventory,close,self.commission)
                account_value = account_profit * self.unit * len(self.inventory)
                avg_value = avg_price * self.unit * len(self.inventory)
                value_diff = (account_value - self.highest_value[0]) / avg_value
                if account_value > self.highest_value[0]: 
                    self.highest_value[0] = account_value
                elif value_diff <= -self.stop_pct and self.highest_value[0] > 0:  #帳面獲利減少的懲罰
                    self.reward = value_diff
                elif account_profit / avg_price < -self.stop_pct:  #帳損超過的懲罰
                    self.reward = account_value / avg_value
                print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
				+ " | Bull: "+ str(len(self.inventory) * self.unit) + ' | Potential: ' + formatPrice(account_value)
				+ " | Reward: " + str(round(self.reward,2)))	
                self.highest_value[1] = 0
            else:
                account_profit, avg_price, avg_close = get_short_account(self.inventory,close,self.commission)
                account_value = account_profit * self.unit * len(self.inventory)
                avg_value = avg_price * self.unit * len(self.inventory)
                value_diff = (account_value - self.highest_value[1]) / avg_value
                if account_value > self.highest_value[1]: 
                    self.highest_value[1] = account_value
                elif value_diff <= -self.stop_pct and self.highest_value[1] > 0:  #帳面獲利減少的懲罰
                    self.reward = value_diff
                elif account_value / avg_value < -self.stop_pct:  #帳損超過的懲罰
                    self.reward = account_value / avg_value
                print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
				+ " | Bear: "+ str(len(self.inventory) * self.unit) + ' | Potential: ' + formatPrice(account_value)
				+ " | Reward: " + str(round(self.reward,2)))
                self.highest_value[0] = 0
        if len(self.inventory) == 0:
            print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
		    + " | Nuetrual")
            self.highest_value[:] = 0
    
    #用買進動作平倉
    def _long_clean(self, close, e, episode_count, t, l):
        price = close * (1+self.commission)
        sold_price = self.inventory.pop(0)
        profit = (sold_price[1] - price) * self.unit
        self.total_profit += profit
        self.reward = self.pr_ratio * profit / (sold_price[0] * self.unit)
        self.cash += profit + sold_price[0] * self.unit
        print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
		+ " | Bear: "+ str(len(self.inventory) * self.unit) +" | Long: " + formatPrice(price) 
		+ " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(self.total_profit)
		+ " | Reward: " + str(round(self.reward,2)))
            
    def _long_new(self, close, e, episode_count, t, l):
        account_profit, avg_price, avg_close = get_long_account(self.inventory,close,self.commission)
        account_value = account_profit * self.unit * len(self.inventory)
        avg_value = avg_price * self.unit * len(self.inventory)
        value_diff = (account_value - self.highest_value[0]) / avg_value
        if account_value > self.highest_value[0]: 
            self.highest_value[0] = account_value
        elif value_diff <= -self.stop_pct and self.highest_value[0] > 0:  #帳面獲利減少的懲罰
            self.reward = value_diff
        elif account_profit / avg_price < -self.stop_pct:  #帳損超過的懲罰
            self.reward = account_value / avg_value
        price = close * (1+self.commission)
        cost = close * self.unit
        self.cash -= cost
        self.inventory.append([close, price, 'long']) #存入進場資訊
        print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
		+ " | Bull: "+ str(len(self.inventory) * self.unit) + " | Long : " + formatPrice(price)
        + ' | Potential: ' + formatPrice(account_value)
		+ " | Reward: " + str(round(self.reward,2)))

    def _long_new_empty(self, close, e, episode_count, t, l):
        price = close * (1+self.commission)
        cost = close * self.unit
        self.cash -= cost
        self.inventory.append([close, price, 'long']) #存入進場資訊
        print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
		+ " | Bull: "+ str(len(self.inventory) * self.unit) + " | Long : " + formatPrice(price))

    #用賣出的動作來平倉
    def _short_clean(self, close, e, episode_count, t, l):
        price = close * (1-self.commission)
        bought_price = self.inventory.pop(0)
        profit = (price - bought_price[1]) * self.unit
        self.total_profit += profit
        self.reward = self.pr_ratio * profit / (bought_price[0] * self.unit)
        self.cash += profit + bought_price[0] * self.unit
        print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
		+ " | Bull: "+ str(len(self.inventory) * self.unit) +" | Short: " + formatPrice(price) 
		+ " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(self.total_profit)
		+ " | Reward: " + str(round(self.reward,2)))

    def _short_new(self, close, e, episode_count, t, l):
        account_profit, avg_price, avg_close = get_short_account(self.inventory,close,self.commission)
        account_value = account_profit * self.unit * len(self.inventory)
        avg_value = avg_price * self.unit * len(self.inventory)
        value_diff = (account_value - self.highest_value[1]) / avg_value
        if account_value > self.highest_value[1]: 
            self.highest_value[1] = account_value
        elif value_diff <= -self.stop_pct and self.highest_value[1] > 0:  #帳面獲利減少的懲罰
            self.reward = value_diff
        elif account_value / avg_value < -self.stop_pct:  #帳損超過的懲罰
            self.reward = account_value / avg_value
        price = close * (1-self.commission)
        cost = close * self.unit #做空一樣要付出成本，保證金的概念
        self.cash -= cost
        self.inventory.append([close, price, 'short']) #存入進場資訊
        print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
		+ " | Bear: "+ str(len(self.inventory) * self.unit) + " | Short : " + formatPrice(price)
        + ' | Potential: ' + formatPrice(account_value)
		+ " | Reward: " + str(round(self.reward,2)))
    
    def _short_new_empty(self, close, e, episode_count, t, l):
        price = close * (1 - self.commission)
        cost = close * self.unit #做空一樣要付出成本，保證金的概念
        self.cash -= cost
        self.inventory.append([close, price,'short']) #存入進場資訊
        print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
		+ " | Bear: "+ str(len(self.inventory) * self.unit) + " | Short : " + formatPrice(price))
    
    #全部平倉
    def _clean_inventory(self, close, e, episode_count, t, l): 
        if  self.inventory[0][-1] == 'long':
            account_profit, avg_price, avg_close = get_long_account(self.inventory,close,self.commission)
            self.reward = self.pr_ratio * (account_profit / avg_price) * len(self.inventory)
            profit = account_profit * self.unit * len(self.inventory)
            self.total_profit += profit
            self.cash += avg_close * len(self.inventory) * self.unit + profit
            print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
				+ " | Clean Inventory"+ ' | Profit: ' + formatPrice(profit)
				+ " | Total Profit: " + formatPrice(self.total_profit)
				+ " | Reward: " + str(round(self.reward,2)))
        elif self.inventory[0][-1] == 'short':
            account_profit, avg_price, avg_close = get_short_account(self.inventory,close,self.commission)
            self.reward = self.pr_ratio * (account_profit / avg_price) * len(self.inventory)
            profit = account_profit * self.unit * len(self.inventory)
            self.total_profit += profit
            self.cash += avg_close * len(self.inventory) * self.unit + profit
            print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(self.cash)
				+ " | Clean Inventory"+ ' | Profit: ' + formatPrice(profit)
				+ " | Total Profit: " + formatPrice(self.total_profit)
				+ " | Reward: " + str(round(self.reward,2)))
        self.inventory = [] 
        self.highest_value[:] = 0
				

		