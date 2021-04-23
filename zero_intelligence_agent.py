import numpy as np
import math
from agent import Agent

class ZeroIntelligenceAgent(Agent):
    num_of_agent = 0
    
    def __init__(self, start_cash = 1000000, bid_side = 0.8, trading_time = 10, range_of_price = 0.01, range_of_quantity = 5):
        super().__init__('ZERO_INTELLIGENCE', start_cash)
        ZeroIntelligenceAgent.add_counter()

        self.trading_time = trading_time
        self.range_of_quantity = range_of_quantity
        self.range_of_price = range_of_price

    def step(self):
        super().step()
        # to trade?
        if np.random.binomial(n = 1, p = 0.5) == 1:
            order = self.genertate_order()
            self.place_order('limit', order['code'], order['bid_or_ask'], order['quantity'], order['price'])
        return    
    def genertate_order(self):
        # which one?
        code = np.random.choice(self.order.keys())
        # if existed, modify the order
        price_info = self.core.get_price_info(code)

        if np.random.binomial(n = 1, p = 0.8) == 1 or self.holdings[code] == 0:
            bid_or_ask = 'BID'
            quantity = np.random.randint(1, self.range_of_quantity)
            best_price = self.current_price('ASK', code, 1)[0]['price']
        else:
            bid_or_ask = 'ASK'
            quantity = self.holdings[code]
            best_price = self.current_price('BID', code, 1)[0]['price']
        
        tick_range = math.floor((best_price * self.range_of_price) / price_info['tick_size'])
        price = np.clip(best_price + price_info['tick_size'] * np.random.randint(-tick_range, tick_range), price_info['low_limit'], price_info['up_limit'])

        return {'code': code, 'bid_or_ask': bid_or_ask, 'price': price, 'quantity': quantity}
        