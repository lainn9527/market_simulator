import math
import numpy as np
import talib
from message import Message
from order import LimitOrder, MarketOrder
from datetime import datetime, timedelta, time
from typing import Dict, List
class Agent:
    num_of_agent = 0

    def __init__(self, _type, start_cash = 10000000, start_securities = None):
        Agent.add_counter()
        self.type = _type
        self.start_cash = start_cash
        self.unique_id = f"{self.type}_{self.get_counter()}"
        self.core = None
        
        
        '''
        Information of markets
        orders: {security code: [order_id] }
        holdings: {security code: volume}
        '''
        self.cash = start_cash
        self.holdings = start_securities
        self.orders = {code: [] for code in self.holdings.keys()}
        self.orders_history = {code: [] for code in self.holdings.keys()}

        # state flag
        self.is_trading = False
        self.total_timestep = None
    
    def start(self, core):
        self.core = core
        # ready to go

    def step(self):
        pass

    def receive_message(self, message):
        if message.receiver != self.unique_id and message.receiver != 'agents':
            raise Exception('Wrong receiver!')

        message_subject = ['OPEN_SESSION', 'CLOSE_SESSION', 'OPEN_AUCTION', 'CLOSE_AUCTION', 'OPEN_CONTINUOUS_TRADING', 'STOP_CONTINUOUS_TRADING', 'ORDER_PLACED', 'ORDER_CANCELLED', 'ORDER_INVALIDED', 'ORDER_FILLED', 'ORDER_FINISHED']

        if message.subject == 'OPEN_SESSION':
            # receive time and ready to palce order
            self.handle_open()

        elif message.subject == 'CLOSE_SESSION':
            # receive the daily info of market: daily_info[code] = {'date': date in isoformat, 'order': self.orders, 'bid': self.bids, 'ask': self.asks, 'stats': self.stats }
            self.handle_close()

        elif message.subject == 'ORDER_PLACED':
            # check if the order is transformed from market order
            self.orders[message.content['code']].append(message.content['order_id'])            
            # self.log_event('ORDER_PLACED', self.orders[order.code][order.order_id])


        elif message.subject == 'ORDER_FILLED':
            order_record = self.core.get_order_record(message.content['code'], message.content['order_id'])
            if message.content['bid_or_ask'] == 'BID':
                self.holdings[message.content['code']] += message.content['quantity']
                self.cash += (order_record.order.price - message.content['price']) * message.content['quantity']
            
            elif order_record.order.bid_or_ask == 'ASK':
                self.cash += message.content['price'] * message.content['quantity']            
            
            self.log_event('ORDER_FILLED', {'price': message.content['price'], 'quantity': message.content['quantity']})


        elif message.subject == 'ORDER_FINISHED':
            for i, order_id in enumerate(self.orders[message.content['code']]):
                if order_id == message.content['order_id']:
                    self.orders[message.content['code']].pop(i)
            self.orders_history[message.content['code']].append(message.content['order_id'])

            self.log_event('ORDER_FINISHED', {'order_record': self.core.get_order_record(code = message.content['code'] , order_id = message.content['order_id'])} )
        
        elif message.subject == 'ORDER_INVALIDED':
            pass
        else:
            raise Exception(f"Invalid subject for agent {self.unique_id}")

    def place_limit_bid_order(self, code, volume, price):
        if volume == 0:
            return
        
        cost = volume * price
        if cost > self.cash:
            return

        self.cash -= cost
        order = LimitOrder(self.unique_id, code, 'LIMIT', 'BID', volume, price)
        msg = Message('MARKET', 'LIMIT_ORDER', self.unique_id, 'market', {'order': order})
        
        self.core.send_message(msg)

    def place_limit_ask_order(self, code, volume, price):
        if volume == 0:
            return

        if volume > self.holdings[code]:
            raise Exception(f"Not enough {code} shares")
        self.holdings[code] -= volume
        order = LimitOrder(self.unique_id, code, 'LIMIT', 'ASK', volume, price)
        msg = Message('MARKET', 'LIMIT_ORDER', self.unique_id, 'market', {'order': order})
        
        self.core.send_message(msg)



    def place_market_bid_order(self, code, volume):
        pass

    def place_market_ask_order(self, code, volume):
        pass
    
    def modify_order(self):
        pass

    def cancel_order(self):
        pass


    def log_event(self, event_type, event):
        # print(f"Agent: {self.unique_id}, Event type: {event_type}, Event: {event}")
        return
    
    @classmethod
    def add_counter(cls):
        cls.num_of_agent += 1

    @classmethod
    def get_counter(cls):
        return cls.num_of_agent
    
    def generate_id(self):
        pass
        
    def handle_open(self):
        pass

    def schedule_auction(self):
        pass
    
    def schedule_continuous_trading(self):
        pass
    
    def handle_close(self):
        pass

class TestAgent(Agent):
    num_of_agent = 0
    
    def __init__(self, order_list, start_cash = 1000000, start_securities = None):
        super().__init__('TEST', start_cash, start_securities)
        TestAgent.add_counter()
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

class ZeroIntelligenceAgent(Agent):
    num_of_agent = 0
    
    def __init__(self, start_cash = 1000000, start_securities = None, bid_side = 0.8, range_of_price = 5, range_of_quantity = 5):
        super().__init__('ZERO_INTELLIGENCE', start_cash, start_securities)
        ZeroIntelligenceAgent.add_counter()
        self.range_of_quantity = range_of_quantity
        self.range_of_price = range_of_price

    def step(self):
        super().step()
        # to trade?
        if np.random.binomial(n = 1, p = 0.01) == 1:
            self.generate_order()

    def generate_order(self):
        # which one?
        code = np.random.choice(list(self.holdings.keys()))
        current_price = self.core.get_current_price(code)
        quantity = np.random.randint(1, self.range_of_quantity)
        tick_size = self.core.get_tick_size(code)

        # TODO: best bid and best ask
        if np.random.binomial(n = 1, p = 0.5) == 1 or self.holdings[code] == 0:
            bid_or_ask = 'BID'
            # lowest_ask = self.core.
            price = round(current_price + np.random.randint(1, self.range_of_price) * tick_size, 2)
            self.place_limit_bid_order(code, quantity, price)
        else:
            bid_or_ask = 'ASK'
            price = round(current_price - np.random.randint(1, self.range_of_price) * tick_size, 2)
            self.place_limit_ask_order(code, min(quantity, self.holdings[code]), price)
        # if existed, modify the order
        # if len(self.orders[code]) != 0:
            # pass

class ChartistAgent(Agent):
    num_of_agent = 0
    def __init__(self, start_cash: int = 1000000, start_securities: Dict[str, int] = None, strategy = None, risk_preference = 1):
        super().__init__('ChartistAgent', start_cash, start_securities)
        ChartistAgent.add_counter()
        self.strategy = strategy
        self.risk_preference = risk_preference
        self.cool_down = 0

    def step(self):
        if self.cool_down > 0:
            self.cool_down -= 1
            return
        default_cool_down = np.random.randint(500, 1000)
        default_bid_signal = 1.05
        default_ask_signal = 0.95

        code = "TSMC"
        price_list = self.core.get_records(code = code, _type = 'average', step = 100)
        if len(price_list) < 100:
            return
        
        if self.sma(price_list[:10]) / self.sma(price_list[:30]) >= 1.05:
            if np.random.binomial(n = 1, p = 0.5 * self.risk_preference) == 1:
                self.generate_bid_order(code, amount = 0.1 * self.cash)
            self.cool_down = round(default_cool_down / self.risk_preference)
        
        elif self.sma(price_list[:10]) / self.sma(price_list[:30]) <= 0.95:
            if np.random.binomial(n = 1, p = 0.5 / self.risk_preference) == 1:
                self.generate_ask_order(code, quantity = round(0.3 * self.holdings[code]))
            self.cool_down = round(default_cool_down / self.risk_preference)
            
    def generate_bid_order(self, code, amount):
        tick_size = self.core.get_tick_size(code)
        price = round(self.core.get_current_price(code) + 2 * tick_size, 2)
        quantity = amount // price
        self.place_limit_bid_order(code, volume = quantity, price = price)
        
    def generate_ask_order(self, code, quantity):
        tick_size = self.core.get_tick_size(code)
        price = round(self.core.get_current_price(code) - 2 * tick_size, 2)
        self.place_limit_ask_order(code, volume = quantity, price = price)

    def sma(self, price_list):
        return sum(price_list) / len(price_list)

    def ema(self):
        pass
    def macd(self):
        pass
    

class FundamentalistAgent(Agent):
    pass

class BrokerAgent(Agent):
    num_of_agent = 0
    def __init__(self, code, start_cash = 200000000, target_volume = 500):
        super().__init__('BROKER', start_cash)
        BrokerAgent.add_counter()

        # responsible for creating the liquidity of certain security
        self.code = code
        self.target_volume = target_volume
        self.price_index = {}
    
    
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
