from functions import *
from trading import Trading

class Profolio():
    def __init__(self,init_cash):
        self.max_drawdown = 0
        self.init_cash = init_cash
        self.profolio_value = init_cash

    def return_ratio(self, total_profit):
        return round(100 * total_profit / self.init_cash, 2)

    def total_value(self, unit, close, cash, inventory, commission):
        if len(inventory) > 0:
            inventory_value = get_inventory_value(inventory, close, commission)
            inventory_value *= unit
            self.profolio_value = inventory_value + cash
        else:
            self.profolio_value = cash

    def eval_draw_down(self, unit, close, cash, inventory, commission):
        self.total_value(unit, close, cash, inventory,commission)
        if self.profolio_value - self.init_cash < 0:  #虧損時才做
            drawdown = (self.profolio_value - self.init_cash) / self.init_cash
            if drawdown < self.max_drawdown:
                self.max_drawdown = drawdown

