import numpy as np
import math
from agent import Agent

class BrokerAgent(Agent):
    num_of_agent = 0
    def __init__(self, _type, code, _id = None, start_cash = 200000000, security_unit = 1000, target_volume = 500):
        super().__init__(_type, _id, start_cash, security_unit)
        BrokerAgent.add_counter()

        # responsible for creating the liquidity of certain security
        self.code = code
        self.target_volume = target_volume
    def step(self):
        super().step()

    def schedule_auction(self):
        # provide liquidity when open session
        if self.is_open_auction == True and self.is_trading == True:
            best_ask = self.current_price("ASK", self.code, 1)[0]['price']
            price_info = self.core.get_price_info(self.code)
            price_list = [best_ask + price_info['tick_size'] * i for i in range(-5, 5)]
            placed_volumn = 0
            while placed_volumn < self.target_volume:
                volumn = np.random.randint(self.target_volume // 20, self.target_volume // 10)
                self.place_order('auction', self.code, 'ASK', volumn, np.random.choice(price_list))


    def schedule_continuous_trading(self):
        pass


            
            
            

