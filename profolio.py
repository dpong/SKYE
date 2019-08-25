from functions import *
from trading import Trading
import numpy as np
import math

class Profolio():
    def __init__(self,init_cash):
        self.max_drawdown = 0
        self.init_cash = init_cash
        self.profolio_value = init_cash

    def total_value(self, unit, close, cash, inventory, commission):
        if len(inventory) > 0:
            inventory_value = get_inventory_value(inventory, unit, close, commission)
            self.profolio_value = inventory_value + cash
        else:
            self.profolio_value = cash

    def eval_draw_down(self, unit, close, cash, inventory, commission):
        self.total_value(unit, close, cash, inventory,commission)
        if self.profolio_value - self.init_cash < 0:  #虧損時才做
            drawdown = (self.profolio_value - self.init_cash) / self.init_cash
            if drawdown < self.max_drawdown:
                self.max_drawdown = drawdown

    def sharp_ratio(self, data):
        interest = 2  #無風險利率(%)
        m1 = data[0,-2] + data[0,-1]  # cash + holding = profolio value
        last = data[-1,-2] + data[-1,-1]
        return_ratio = (last - m1) / m1 * 100
        l = []
        for i in range(len(data)-1):
            m2 = data[i+1,-2] + data[i+1,-1]
            l.append((m2 - m1)/ m1 * 100)
            m1 = m2
        l = np.array(l)
        std = np.std(l)
        if std == 0:
            sharp = 0
        else:
            sharp = (return_ratio - interest) / std 
        sharp = sharp / math.sqrt(len(data)) * math.sqrt(252)  #年化
        return round(sharp,2)

    