import math
import numpy as np
import talib
from message import Message
from order import LimitOrder, MarketOrder
from datetime import datetime, timedelta, time
from typing import Dict, List
class Agent:
    num_of_agent = 0

    def __init__(self, _type, start_cash = 10000000, start_securities = None, risk_preference = 1):
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
        self.reserved_cash = 0
        self.reserved_holdings = {code: 0 for code in self.holdings.keys()}
        self.wealth = 0
        self.orders = {code: [] for code in self.holdings.keys()}
        self.orders_history = {code: [] for code in self.holdings.keys()}

        # state flag
        self.risk_preference = risk_preference
        
    
    def start(self, core):
        self.core = core
        

    def step(self):
        self.update_wealth()


    def receive_message(self, message):
        if message.receiver != self.unique_id and message.receiver != 'agents':
            raise Exception('Wrong receiver!')

        message_subject = ['OPEN_SESSION', 'CLOSE_SESSION', 'OPEN_AUCTION', 'CLOSE_AUCTION', 'OPEN_CONTINUOUS_TRADING', 'STOP_CONTINUOUS_TRADING', 'ORDER_PLACED', 'ORDER_CANCELLED', 'ORDER_INVALIDED', 'ORDER_FILLED', 'ORDER_FINISHED', 'ISSUE_INTEREST']

        if message.subject == 'OPEN_SESSION':
            # receive time and ready to palce order
            self.handle_open()

        elif message.subject == 'ORDER_PLACED':
            # check if the order is transformed from market order
            self.orders[message.content['code']].append(message.content['order_id'])            
            # self.log_event('ORDER_PLACED', self.orders[order.code][order.order_id])


        elif message.subject == 'ORDER_FILLED':         
            self.handle_filled_order(message.content['code'], message.content['order_id'], message.content['price'], message.content['quantity'])


        elif message.subject == 'ORDER_FINISHED':
            self.handle_finished_order(message.content['code'], message.content['order_id'])
        
        elif message.subject == 'ORDER_CANCELLED':
            self.handle_cancelled_order(code = message.content['code'], order_id = message.content['order_id'], unfilled_amount = message.content['unfilled_amount'], unfilled_quantity = message.content['unfilled_quantity'])


        elif message.subject == 'ISSUE_INTEREST':
            interest_rate = message.content['interest_rate']
            self.cash += round(self.cash * interest_rate, 2)
        elif message_subject == 'ISSUE_DIVIDEND':
            dividend = self.market.stock_size * message.content['dividend']
            self.cash += dividend

        elif message.subject == 'ORDER_INVALIDED':
            pass
        else:
            raise Exception(f"Invalid subject for agent {self.unique_id}")

    def place_limit_bid_order(self, code, volume, price):
        price = round(price, 2)
        cost = volume * price * 100
        if price <= 0 or volume == 0 or cost > self.cash:
            return
        if price > 500:
            self.for_break()

        self.cash -= cost
        self.reserved_cash += cost
        order = LimitOrder(self.unique_id, code, 'LIMIT', 'BID', volume, price)
        msg = Message('MARKET', 'LIMIT_ORDER', self.unique_id, 'market', {'order': order})
        
        self.core.send_message(msg)

    def place_limit_ask_order(self, code, volume, price):
        price = round(price, 2)
        if volume == 0 or price <= 0:
            return
        if volume > self.holdings[code]:
            raise Exception(f"Not enough {code} shares")

        self.holdings[code] -= volume
        self.reserved_holdings[code] += volume
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

    def handle_filled_order(self, code, order_id, price, quantity):
        order_record = self.core.get_order_record(code, order_id)
        if order_record.order.bid_or_ask == 'BID':
            self.holdings[code] += quantity
            self.cash += (order_record.order.price - price) * quantity * 100
            self.reserved_cash -= order_record.order.price * quantity * 100
        
        elif order_record.order.bid_or_ask == 'ASK':
            self.cash += price * quantity * 100
            self.reserved_holdings[code] -= quantity

        self.log_event('ORDER_FILLED', {'price': price, 'quantity': quantity})

    def handle_finished_order(self, code, order_id):
        self.orders[code].remove(order_id)
        self.orders_history[code].append(order_id)
        self.log_event('ORDER_FINISHED', {'order_record': self.core.get_order_record(code = code , order_id = order_id)})


    def handle_cancelled_order(self, code, order_id, unfilled_amount, unfilled_quantity):
        order_record = self.core.get_order_record(code, order_id)
        if order_record.order.bid_or_ask == 'BID':
            self.cash += unfilled_amount
            self.reserved_cash -= unfilled_amount
        elif order_record.order.bid_or_ask == 'ASK':
            self.holdings[code] += unfilled_quantity
            self.reserved_holdings[code] -= unfilled_quantity
            
        self.orders[code].remove(order_id)
        self.orders_history[code].append(order_id)

    def update_wealth(self):
        cash = self.cash + self.reserved_cash
        securities_value = 0
        for code in self.holdings.keys():
            average_price = self.core.get_records(code, 'average', step = 1)[0] if self.get_time() > 0 else self.core.get_current_price(code)
            securities_value +=  average_price * (self.holdings[code] + self.reserved_holdings[code])        
        self.wealth = cash + securities_value

    def get_time(self):
        return self.core.timestep

    def log_event(self, event_type, event):
        # print(f"Agent: {self.unique_id}, Event type: {event_type}, Event: {event}")
        return
    
    def handle_open(self):
        return

    def for_break(self):
        pass
    @classmethod
    def add_counter(cls):
        cls.num_of_agent += 1

    @classmethod
    def get_counter(cls):
        return cls.num_of_agent
    

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
    
    def __init__(self, start_cash = 1000000, start_securities = None, bid_side = 0.5, range_of_price = 5, range_of_quantity = 5):
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
    def __init__(self, start_cash: int = 1000000, start_securities: Dict[str, int] = None, risk_preference = 0, strategy = None):
        super().__init__('ChartistAgent', start_cash, start_securities,risk_preference)
        ChartistAgent.add_counter()
        self.strategy = strategy
        self.risk_preference = risk_preference
        self.quota = 0
        self.adjust_period = 200
        self.cool_down = 0
    
    def step(self):
        super().step()
        if self.get_time() // self.adjust_period == 0:
            self.adjust_portfolio()

        if self.quota < 0:
            return

        elif self.cool_down > 0:
            self.cool_down -= 1
            return

        default_cool_down = np.random.randint(50, 150)
        self.generate_order()
    
    def generate_order(self):
        code = "TSMC"
        time_window = self.strategy['time_window']
        current_price = self.core.get_current_price(code)
        price_list = self.core.get_records(code = code, _type = 'average', step = time_window)

        if len(price_list) < time_window:
            return

        trend = current_price - self.sma(price_list)
        bid_amount = self.risk_preference * 0.3 * self.quota
        ask_quantity = round(self.risk_preference * 0.3 * self.holdings[code])
        price = current_price + trend
        
        if trend > 0:
            self.generate_bid_order(code, quantity = round(bid_amount / price), price = price)

        elif trend < 0:
            self.generate_ask_order(code, quantity = ask_quantity, price = price)

    def generate_bid_order(self, code, quantity, price = None):
        tick_size = self.core.get_tick_size(code)
        price = price if price != None else round(self.core.get_current_price(code) + 2 * tick_size, 2)
        self.place_limit_bid_order(code, volume = quantity, price = price)
        
    def generate_ask_order(self, code, quantity, price = None):
        tick_size = self.core.get_tick_size(code)
        price = price if price != None else round(self.core.get_current_price(code) - 2 * tick_size, 2)
        self.place_limit_ask_order(code, volume = quantity, price = price)

    def adjust_portfolio(self):
        self.quota = self.risk_preference * self.wealth


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
        if self.is_open_auction == True:
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
