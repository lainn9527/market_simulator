import numpy as np
import math
from .agent import Agent

class ZeroIntelligenceAgent(Agent):
    num_of_agent = 0
    
    def __init__(self, start_cash = 1000000, start_securities = None, bid_side = 0.8, range_of_price = 5, range_of_quantity = 5):
        super().__init__('ZERO_INTELLIGENCE', start_cash)
        ZeroIntelligenceAgent.add_counter()
        self.holdings.update(start_securities)
        self.range_of_quantity = range_of_quantity
        self.range_of_price = range_of_price

    def step(self):
        super().step()
        # to trade?
        if np.random.binomial(n = 1, p = 0.01) == 1:
            self.generate_order()

    def generate_order(self):
        # which one?
        code = np.random.choice(self.security_list)
        current_price = self.core.get_current_price(code)
        quantity = np.random.randint(1, self.range_of_quantity)
        tick_size = self.core.get_tick_size(code)

        if np.random.binomial(n = 1, p = 0.5) == 1 or self.holdings[code] == 0:
            bid_or_ask = 'BID'
            price = round(current_price + np.random.randint(1, self.range_of_price) * tick_size, 2) 
            self.place_limit_bid_order(code, quantity, price)
        else:
            bid_or_ask = 'ASK'
            price = round(current_price - np.random.randint(1, self.range_of_price) * tick_size, 2)
            self.place_limit_ask_order(code, min(quantity, self.holdings[code]), price)
        # if existed, modify the order
        # if len(self.orders[code]) != 0:
            # pass

        