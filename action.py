from functions import *

class Action():
    def __init__(self,unit):
        self.safe_margin = 0.8  #現金安全水位
        self.unit = unit
        self.highest_value = np.array([0,0])  #0是多倉位，1是空倉位
        self.reward = 0
        self.commission = 0.003  #千分之三的手續費
        self.stop_pct = 0.1  #停損%數

    def _hold(self,close,cash,inventory,e,episode_count,t,l):
        if len(inventory) > 0:
            if inventory[0][1] == 'long':
                account_profit, avg_price = get_long_account(inventory,close,self.commission)
                account_value = account_profit * self.unit * len(inventory)
                avg_value = avg_price * self.unit * len(inventory)
                value_diff = (account_value - self.highest_value[0]) / avg_value
                if account_value > self.highest_value[0]: 
                    self.highest_value[0] = account_value
                elif value_diff <= -self.stop_pct and self.highest_value[0] > 0:  #帳面獲利減少的懲罰
                    self.reward = value_diff
                elif account_profit / avg_price < -self.stop_pct:  #帳損超過的懲罰
                    self.reward = account_value / avg_value
                print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
				+ " | Bull: "+ str(len(inventory) * self.unit) + ' | Potential: ' + formatPrice(account_value)
				+ " | Highest: " + str(round(self.highest_value[0],2))
				+ " | Reward: " + str(round(self.reward,2)))	
                self.highest_value[1] = 0
            else:
                account_profit, avg_price = get_short_account(inventory,close,self.commission)
                account_value = account_profit * self.unit * len(inventory)
                avg_value = avg_price * self.unit * len(inventory)
                value_diff = (account_value - self.highest_value[1]) / avg_value
                if account_value > self.highest_value[1]: 
                    self.highest_value[1] = account_value
                elif value_diff <= -self.stop_pct and self.highest_value[1] > 0:  #帳面獲利減少的懲罰
                    self.reward = value_diff
                elif account_value / avg_value < -self.stop_pct:  #帳損超過的懲罰
                    self.reward = account_value / avg_value
                print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
				+ " | Bear: "+ str(len(inventory) * self.unit) + ' | Potential: ' + formatPrice(account_value)
				+ " | Highest: " + str(round(self.highest_value[1],2))
				+ " | Reward: " + str(round(self.reward,2)))
                self.highest_value[0] = 0
        if len(inventory) == 0:
            print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
		    + " | Nuetrual")
            self.highest_value[:] = 0
    

    def _long_clean(self,close,cash,inventory,total_profit,e,episode_count,t,l):
        price = close * (1+self.commission)
        sold_price = inventory.pop(0)
        profit = (sold_price[0] - price) * self.unit
        total_profit += profit
        self.reward = profit / (sold_price[0] * self.unit)
        cash += profit + sold_price[0] * self.unit
        print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
		+ " | Bear: "+ str(len(inventory) * self.unit) +" | Long: " + formatPrice(price) 
		+ " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(total_profit)
		+ " | Reward: " + str(round(self.reward,2)))
            
        return cash, inventory, total_profit

    def _long_new(self,close,cash,inventory,total_profit,e,episode_count,t,l):
        price = close * (1+self.commission)
        cost = price * self.unit
        cash -= cost
        inventory.append([price,'long']) #存入進場資訊
        print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
		+ " | Bull: "+ str(len(inventory) * self.unit) + " | Long : " + formatPrice(price))
        
        return cash, inventory, total_profit

    def _short_clean(self,close,cash,inventory,total_profit,e,episode_count,t,l):
        price = close * (1-self.commission)
        bought_price = inventory.pop(0)
        profit = (price - bought_price[0]) * self.unit
        total_profit += profit
        self.reward = profit / (bought_price[0] * self.unit)
        cash += profit + bought_price[0] * self.unit
        print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
		+ " | Bull: "+ str(len(inventory) * self.unit) +" | Short: " + formatPrice(price) 
		+ " | Profit: " + formatPrice(profit)+ " | Total Profit: " + formatPrice(total_profit)
		+ " | Reward: " + str(round(self.reward,2)))

        return cash, inventory, total_profit

    def _short_new(self,close,cash,inventory,total_profit,e,episode_count,t,l):
        price = close * (1-self.commission)
        cost = close * self.unit #做空一樣要付出成本，保證金的概念
        cash -= cost
        inventory.append([price,'short']) #存入進場資訊
        print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
		+ " | Bear: "+ str(len(inventory) * self.unit) + " | Short : " + formatPrice(price))

        return cash, inventory, total_profit

    def _clean_inventory(self,close,cash,inventory,total_profit,e,episode_count,t,l):
        if  inventory[0][1] == 'long':
            account_profit, avg_price = get_long_account(inventory,close,self.commission)
            self.reward = (account_profit / avg_price) * len(inventory)
            profit = account_profit * self.unit * len(inventory)
            total_profit += profit
            cash += avg_price * len(inventory) * self.unit + profit
            print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
				+ " | Clean Inventory"+ ' | Profit: ' + formatPrice(profit)
				+ " | Total Profit: " + formatPrice(total_profit)
				+ " | Reward: " + str(round(self.reward,2)))
        elif inventory[0][1] == 'short':
            account_profit, avg_price = get_short_account(inventory,close,self.commission)
            self.reward = (account_profit / avg_price) * len(inventory)
            profit = account_profit * self.unit * len(inventory)
            total_profit += profit
            cash += avg_price * len(inventory) * self.unit + profit
            print("Ep " + str(e) + "/" + str(episode_count)+" %.2f%%" % round(t*(100/l),2) + " Cash: " + formatPrice(cash)
				+ " | Clean Inventory"+ ' | Profit: ' + formatPrice(profit)
				+ " | Total Profit: " + formatPrice(total_profit)
				+ " | Reward: " + str(round(self.reward,2)))
        inventory = [] #全部平倉
        self.highest_value[:] = 0
        return cash, inventory, total_profit
				

		