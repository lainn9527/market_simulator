import numpy as np
import math
from .agent import Agent

class TestAgent(Agent):
    num_of_agent = 0
    
    def __init__(self, order_list, start_cash = 1000000, start_securities = {'TSMC': 100}):
        super().__init__('TEST', start_cash)
        TestAgent.add_counter()
        self.holdings.update(start_securities)
        self.order_list = order_list

    def step(self):
        super().step()
        # to trade?
        if len(self.order_list) != 0:
            order = self.order_list.pop(0)
            if order['bid_or_ask'] == 'BID':
                self.place_limit_bid_order(order['code'], order['quantity'], order['price'])
            else:
                self.place_limit_ask_order(order['code'], order['quantity'], order['price'])

        