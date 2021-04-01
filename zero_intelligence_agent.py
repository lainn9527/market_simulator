import numpy as np
from agent import Agent

class ZeroIntelligenceAgent(Agent):
    num_of_agent = 0
    
    def __init__(self, _type, _id = None, start_cash = 1000000, security_unit = 1000, bid_side = 0.8, trading_time = 10, range_of_price = 0.1, range_of_quantity = 5):
        super().__init__(_type, _id, start_cash, security_unit)
        ZeroIntelligenceAgent.add_counter()

        self.bid_side = bid_side
        self.trading_time = trading_time
        self.range_of_quantity = range_of_quantity
        self.range_of_price = range_of_price
        self.wake_up = list()

    def step(self):
        super().step()
        if not self.is_trading or self.current_time < self.wake_up[0]:
            return

        elif self.current_time == self.wake_up[0]:
            order = self.genertate_order()
            if order == None:
                return
            if self.is_open_auction or self.is_close_auction:
                self.place_order('auction', order['code'], order['bid_or_ask'], order['quantity'], order['price'])
            elif self.is_continuous_trading:
                self.place_order('limit', order['code'], order['bid_or_ask'], order['quantity'], order['price'])
            else:
                raise Exception
                    
    def schedule_auction(self):
        # random a moment to place the order
        self.wake_up.append(self.current_time + self.timestep_to_time(np.random.randint(1, self.period_timestep)))
    
    def schedule_continuous_trading(self):
        schedule = np.randomint(1, self.period_timestep, self.trading_time - 1)
        schedule.sort()
        for item in schedule:
            self.wake_up.append(self.current_time + self.timestep_to_time(item))
    
    def genertate_order(self):
        no_order_security = [code for code, orders in self.orders.items() if len(orders) == 0 or orders[-1]['finished_time'] != None]
        if len(no_order_security) == 0:
            return None

        code = np.random.choice(no_order_security)
        if np.random.binomial(n = 1, p = 0.8) == 1 or self.holdings[code] == 0:
            bid_or_ask = 'BID'
            quantity = np.random.randint(1, self.range_of_quantity) * self.security_unit
        else:
            bid_or_ask = 'ASK'
            quantity = self.holdings[code]
        price = (np.random.rand() * (self.range_of_price * 2) - self.range_of_price + 1) * self.current_price(bid_or_ask, code, 1) # -price range ~ +price range

        return {'code': code, 'bid_or_ask': bid_or_ask, 'price': price, 'quantity': quantity}
        