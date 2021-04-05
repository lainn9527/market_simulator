import numpy as np
import math
from agent import Agent

class BrokerAgent(Agent):
    num_of_agent = 0
    def __init__(self, code, _id = None, start_cash = 200000000, security_unit = 1000, target_volume = 500):
        super().__init__('BROKER', _id, start_cash, security_unit)
        BrokerAgent.add_counter()

        # responsible for creating the liquidity of certain security
        self.code = code
        self.target_volume = target_volume * security_unit
        self.price_index = {}
    
    def start(self, core, start_time, time_scale, securities):
        super().start(core, start_time, time_scale, securities)
        self.holdings[self.code] = self.target_volume
    
    def step(self):
        super().step()
    

    def schedule_auction(self):
        # provide liquidity when open session
        if self.code not in self.holdings.keys():
            raise Exception
        if self.is_open_auction == True and self.is_trading == True:
            best_ask = self.current_price("ASK", self.code, 1)[0]['price']
            price_info = self.core.get_price_info(self.code)
            price_list = [best_ask + price_info['tick_size'] * i for i in range(-5, 5)]
            placed_volumn = 0
            order_counter = 0
            for price in price_list:
                volumn = (np.random.randint(self.target_volume // 20, self.target_volume // 10) // 1000) * 1000
                self.price_index[price] = order_counter
                order_counter += 1
                if placed_volumn <= self.target_volume:
                    self.place_order('auction', self.code, 'ASK', volumn, price)
                else:
                    self.place_order('auction', self.code, 'ASK', self.target_volume - placed_volumn, price)
                    break


    def schedule_continuous_trading(self):
        pass