import math
import numpy as np
import random
from numpy.lib.function_base import quantile
import talib
from datetime import datetime, timedelta, time
from typing import Dict, List

from .message import Message
from .order import LimitOrder, MarketOrder

class Agent:
    num_of_agent = 0

    def __init__(self, _id, _type, start_cash = 10000000, start_securities = None, risk_preference = 1):
        Agent.add_counter()
        self.type = _type
        self.unique_id = _id
        self.core = None
        
        
        '''
        Information of markets
        orders: {security code: [order_id] }
        holdings: {security code: volume}
        '''
        self.cash = start_cash
        self.holdings = {code: num for code, num in start_securities.items()}
        self.reserved_cash = 0
        self.reserved_holdings = {code: 0 for code in self.holdings.keys()}
        self.wealth = 0
        self.orders = {code: [] for code in self.holdings.keys()}
        self.orders_history = {code: [] for code in self.holdings.keys()}
        self.initial_wealth = 0
        self.average_cost = 0
        self.bids_volume = 0
        self.asks_volume = 0

        # state flag
        self.risk_preference = risk_preference
        
    
    def start(self, core):
        self.core = core
        self.initial_wealth = self.cash + sum([self.core.get_value(code) * num * self.core.get_stock_size() for code, num in self.holdings.items()])
        self.average_cost = self.core.get_value('TSMC')
        # self.average_cost = round(random.gauss(mu = self.core.get_value('TSMC'), sigma = 10), 1)

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
            self.handle_filled_order(message.content['code'], message.content['order_id'], message.content['price'], message.content['quantity'], message.content['transaction_cost'])


        elif message.subject == 'ORDER_FINISHED':
            self.handle_finished_order(message.content['code'], message.content['order_id'])
        
        elif message.subject == 'ORDER_CANCELLED':
            self.handle_cancelled_order(code = message.content['code'], order_id = message.content['order_id'], refund_cash = message.content['refund_cash'], refund_security = message.content['refund_security'])


        elif message.subject == 'ISSUE_INTEREST':
            interest_rate = message.content['interest_rate']
            self.cash += round(self.cash * interest_rate, 2)
            
        elif message_subject == 'ISSUE_DIVIDEND':
            dividend = self.core.get_stock_size() * message.content['dividend']
            self.cash += dividend

        elif message.subject == 'ORDER_INVALIDED':
            pass
        else:
            raise Exception(f"Invalid subject for agent {self.unique_id}")

    def place_limit_bid_order(self, code, quantity, price):
        tick_size = self.core.get_tick_size(code)
        stock_size = self.core.get_stock_size()
        price = round(tick_size * ((price+1e-6) // tick_size), 2)
        transaction_rate = self.core.get_transaction_rate()
        cost = round(quantity * price * stock_size * (1 + transaction_rate), 2)
        if price <= 0 or quantity == 0 or cost > self.cash:
            return
        if price > 500:
            self.for_break()

        self.cash -= cost
        self.reserved_cash += cost
        self.bids_volume += quantity

        order = LimitOrder(self.unique_id, code, 'LIMIT', 'BID', quantity, price)
        msg = Message('MARKET', 'LIMIT_ORDER', self.unique_id, 'market', {'order': order})
        
        self.core.send_message(msg)
    
    def place_limit_ask_order(self, code, quantity, price):
        tick_size = self.core.get_tick_size(code)
        price = round(tick_size * ((price+1e-6) // tick_size), 2)

        if quantity == 0 or price <= 0:
            return
        if quantity > self.holdings[code]:
            # raise Exception(f"Not enough {code} shares")
            return

        self.holdings[code] -= quantity
        self.reserved_holdings[code] += quantity
        self.asks_volume += quantity

        order = LimitOrder(self.unique_id, code, 'LIMIT', 'ASK', quantity, price)
        msg = Message('MARKET', 'LIMIT_ORDER', self.unique_id, 'market', {'order': order})
        
        self.core.send_message(msg)



    def place_market_bid_order(self, code, quantity):
        pass

    def place_market_ask_order(self, code, quantity):
        pass
    
    def modify_order(self):
        pass

    def cancel_order(self):
        pass

    def handle_filled_order(self, code, order_id, price, quantity, transaction_cost):
        stock_size = self.core.get_stock_size()
        order_record = self.core.get_order_record(code, order_id)
        if order_record.order.bid_or_ask == 'BID':
            self.average_cost = round( (stock_size*(self.average_cost * self.holdings[code] + price * quantity) + transaction_cost) / (stock_size*(self.holdings[code] + quantity)), 2)
            self.holdings[code] += quantity
            self.cash += (order_record.order.price - price) * quantity * stock_size
            self.reserved_cash -= (order_record.order.price * quantity * stock_size + transaction_cost) 
        
        elif order_record.order.bid_or_ask == 'ASK':
            self.cash += (price * quantity * stock_size - transaction_cost)
            self.reserved_holdings[code] -= quantity

        self.log_event('ORDER_FILLED', {'price': price, 'quantity': quantity})

    def handle_finished_order(self, code, order_id):
        self.orders[code].remove(order_id)
        self.orders_history[code].append(order_id)
        self.log_event('ORDER_FINISHED', {'order_record': self.core.get_order_record(code = code , order_id = order_id)})


    def handle_cancelled_order(self, code, order_id, refund_cash, refund_security):
        order_record = self.core.get_order_record(code, order_id)
        if order_record.order.bid_or_ask == 'BID':
            self.cash += refund_cash
            self.reserved_cash -= refund_cash
        elif order_record.order.bid_or_ask == 'ASK':
            self.holdings[code] += refund_security
            self.reserved_holdings[code] -= refund_security
            
        self.orders[code].remove(order_id)
        self.orders_history[code].append(order_id)

    def update_wealth(self):
        cash = self.cash + self.reserved_cash
        securities_value = 0
        for code in self.holdings.keys():
            average_price = self.core.get_records(code, 'average', step = 1)[0] if self.get_time() > 0 else self.core.get_current_price(code)
            securities_value +=  average_price * (self.holdings[code] + self.reserved_holdings[code]) * self.core.get_stock_size() 
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
    
    
class RLAgent(Agent):
    num_of_agent = 0
    VALID_ACTION = 1
    INVALID_ACTION = 2
    HOLD = 0
    def __init__(self, _id, start_cash: int = 1000000, start_securities: Dict[str, int] = None):
        super().__init__(_id, 'rl', start_cash = start_cash, start_securities = start_securities)
        RLAgent.add_counter()
        self.action_status = RLAgent.HOLD
    
    def step(self, action = None):
        super().step()
        if isinstance(action, type(None)):
            return

        bid_or_ask = action[0]
        ticks = action[1]
        volume = action[2] + 1
        current_price = self.core.get_current_price('TSMC')
        tick_size = self.core.get_tick_size('TSMC')

        if bid_or_ask == 2:
            self.action_status = RLAgent.HOLD
            return
        elif bid_or_ask == 0:
            # bid
            self.action_status = RLAgent.INVALID_ACTION
            best_bid = self.core.get_best_bids('TSMC', 1)
            best_bid = current_price if len(best_bid) == 0 else best_bid[0]['price']
            price = round(best_bid + (4-ticks) * tick_size, 2)
            self.place_limit_bid_order('TSMC', volume, price)

        elif bid_or_ask == 1:
            # ask
            self.action_status = RLAgent.INVALID_ACTION
            best_ask = self.core.get_best_asks('TSMC', 1)
            best_ask = current_price if len(best_ask) == 0 else best_ask[0]['price']
            price = round(best_ask + (ticks-4) * tick_size, 2)
            self.place_limit_ask_order('TSMC', volume, price)

    def receive_message(self, message):
        super().receive_message(message)
        if message.subject == 'ORDER_PLACED':
            self.action_status = RLAgent.VALID_ACTION


class ZeroIntelligenceAgent(Agent):
    num_of_agent = 0
    
    def __init__(self, _id, start_cash = 1000000, start_securities = None, bid_side = 0.5, range_of_price = 5, range_of_quantity = 5):
        super().__init__(_id, 'zi', start_cash, start_securities)
        ZeroIntelligenceAgent.add_counter()
        self.range_of_quantity = range_of_quantity
        self.range_of_price = range_of_price
        self.bid_side = bid_side
        # self.time_window = time_window

    def step(self):
        super().step()
        # to trade?
        if np.random.binomial(n = 1, p = 0.1) == 1:
            self.provide_liquidity()
            # if self.get_time() < 100:
        elif np.random.binomial(n = 1, p = 0.1) == 1:
            self.provide_volatility()
            # else:
                # self.generate_order()
    def provide_liquidity(self):
        # which one?
        code = np.random.choice(list(self.holdings.keys()))
        current_price = self.core.get_current_price(code)
        quantity = np.random.randint(1, self.range_of_quantity)
        tick_size = self.core.get_tick_size(code)

        if np.random.binomial(n = 1, p = self.bid_side) == 1 or self.holdings[code] == 0:
            best_bid = self.core.get_best_asks(code, 1)
            best_bid = current_price if len(best_bid) == 0 else best_bid[0]['price']
            price = round(best_bid - np.random.randint(0, self.range_of_price) * tick_size , 2)
            self.place_limit_bid_order(code, quantity, price)
        else:
            # ask
            best_ask = self.core.get_best_bids(code, 1)
            best_ask = current_price if len(best_ask) == 0 else best_ask[0]['price']
            price = round(best_ask + np.random.randint(0, self.range_of_price) * tick_size, 2)
            self.place_limit_ask_order(code, min(quantity, self.holdings[code]), price)

    def provide_volatility(self):
        # which one?
        code = np.random.choice(list(self.holdings.keys()))
        current_price = self.core.get_current_price(code)
        quantity = np.random.randint(1, self.range_of_quantity)
        tick_size = self.core.get_tick_size(code)

        if np.random.binomial(n = 1, p = self.bid_side) == 1 or self.holdings[code] == 0:
            best_bid = self.core.get_best_bids(code, 1)
            best_bid = current_price if len(best_bid) == 0 else best_bid[0]['price']
            price = round(best_bid + np.random.randint(-self.range_of_price, self.range_of_price) * tick_size , 2)
            self.place_limit_bid_order(code, quantity, price)
        else:
            # ask
            best_ask = self.core.get_best_asks(code, 1)
            best_ask = current_price if len(best_ask) == 0 else best_ask[0]['price']
            price = round(best_ask - np.random.randint(-self.range_of_price, self.range_of_price) * tick_size, 2)
            self.place_limit_ask_order(code, min(quantity, self.holdings[code]), price)
    def generate_order(self):
        # which one?
        code = np.random.choice(list(self.holdings.keys()))
        current_price = self.core.get_current_price(code)
        quantity = np.random.randint(1, self.range_of_quantity)
        tick_size = self.core.get_tick_size(code)

        if np.random.binomial(n = 1, p = self.bid_side) == 1 or self.holdings[code] == 0:
            # bid
            best_bid = self.core.get_best_bids(code, 1)
            best_bid = current_price if len(best_bid) == 0 else best_bid[0]['price']
            price = round(best_bid - np.random.randint(0, self.range_of_price) * tick_size , 2)
            self.place_limit_bid_order(code, quantity, price)
        else:
            # ask
            best_ask = self.core.get_best_asks(code, 1)
            best_ask = current_price if len(best_ask) == 0 else best_ask[0]['price']
            price = round(best_ask + np.random.randint(0, self.range_of_price) * tick_size, 2)
            self.place_limit_ask_order(code, min(quantity, self.holdings[code]), price)


class RandomAgent(Agent):
    num_of_agent = 0
    def __init__(self, _id, start_cash = 1000000, start_securities = None, time_window = 50, k = 3.85, mean = 1.01):
        super().__init__(_id, 'ra', start_cash, start_securities)
        RandomAgent.add_counter()
        self.mean = mean
        self.time_window = time_window
        self.k = k
        # self.time_window = time_window

    def step(self):
        super().step()
        # to trade?
        if np.random.binomial(n = 1, p = 0.02) == 1:
            self.generate_order()

    def generate_order(self):
        # which one?
        stock_size = self.core.get_stock_size()
        code = np.random.choice(list(self.holdings.keys()))
        current_price = self.core.get_current_price(code)
        price_list = self.core.get_records(code = code, _type = 'average', step = self.time_window)

        if len(price_list) < self.time_window:
            return

        std = self.k * (self.std_of_log_return(price_list))
        mean = self.mean
        tick_size = self.core.get_tick_size(code)
        trade_propotion = np.random.random()
        # quantity = np.random.randint(1, self.range_of_quantity)

        if np.random.binomial(n = 1, p = 0.5) == 1:
            bid_or_ask = 'BID'
            price = current_price + tick_size * round( (current_price * np.random.normal(mean, std) - current_price) / tick_size)
            quantity = math.floor((trade_propotion * self.cash) / (stock_size*price+1e-6))
            self.place_limit_bid_order(code, quantity, price)
        else:
            bid_or_ask = 'ASK'
            price = current_price + tick_size * round((current_price / np.random.normal(mean, std) - current_price) / tick_size)

            quantity = math.floor(trade_propotion * self.holdings[code])
            self.place_limit_ask_order(code, quantity, price)
        if price > 200:
            self.for_break()

    def std_of_log_return(self, price_list):
        return np.std(np.diff(np.log(price_list)))

class TrendAgent(Agent):
    num_of_agent = 0
    def __init__(self, _id, start_cash: int = 1000000, start_securities: Dict[str, int] = None, risk_preference = 0.5, strategy = None, time_window = None):
        super().__init__(_id, 'tr', start_cash, start_securities,risk_preference)
        TrendAgent.add_counter()
        self.strategy = strategy
        self.time_window = time_window
        self.trading_probability = max(0.02 * risk_preference, 0.01)
    
    def step(self):
        super().step()
        if np.random.binomial(n = 1, p = self.trading_probability) == 1:
            self.generate_order()

    def generate_order(self):
        code = "TSMC"
        stock_size = self.core.get_stock_size()
        tick_size = self.core.get_tick_size(code)
        current_price = self.core.get_current_price(code)
        price_list = self.core.get_records(code = code, _type = 'close', step = self.time_window)

        if len(price_list) < self.time_window:
            return
        
        risk_free_amount = (1-self.risk_preference) * self.wealth
        signal = self.get_signal(price_list)
        if current_price > 104:
            current_price
        # add hard limit to sell
        if self.holdings[code] > 0:
            profit_ratio = (current_price / self.average_cost)
            # realize
            if current_price > self.average_cost:
                realize_probability = profit_ratio * (1-self.risk_preference)
                if realize_probability > 1 or np.random.binomial(n = 1, p = realize_probability) == 1:
                    self.place_limit_ask_order(code, quantity = self.holdings[code], price = current_price)
            # flog
            elif profit_ratio < 0.9:
                # the range of loss is -10% ~ -50%, depend on the risk preference
                flog_probability = (1 - profit_ratio) * 2 * (1 - self.risk_preference)
                if flog_probability > 1 or np.random.binomial(n = 1, p = flog_probability) == 1:
                    self.place_limit_ask_order(code, quantity = self.holdings[code], price = current_price)

        if signal > 0 and self.cash > risk_free_amount:
            best_bid = self.core.get_best_bids('TSMC', 1)
            price = round(current_price + np.random.randint(-5, 1) * tick_size , 2)
            bid_quantity = round( (self.cash - risk_free_amount) / (stock_size*price))
            self.place_limit_bid_order(code, quantity = bid_quantity, price = price)

        elif signal < 0 and self.cash < risk_free_amount and self.holdings[code] > 0:
            best_ask = self.core.get_best_asks('TSMC', 1)
            price = round(current_price + np.random.randint(-1, 5) * tick_size , 2)
            ask_quantity = min(round( (risk_free_amount - self.cash) / (stock_size*price)), self.holdings[code])
            self.place_limit_ask_order(code, quantity = ask_quantity, price = price)

    def get_signal(self, price_list, volumn_list = None):
        price_list = np.array(price_list, dtype='float64')
        if self.strategy == 'sma':
            time_window_short = self.time_window // 2
            long = talib.SMA(price_list, self.time_window - 3)
            short = talib.SMA(price_list, time_window_short - 3)
            return float( (short[-3:] - long[-3:]).mean())

        elif self.strategy == 'ema':
            time_window_short = self.time_window // 2
            long = talib.EMA(price_list, self.time_window - 3)
            short = talib.EMA(price_list, time_window_short - 3)
            return float( (short[-3:] - long[-3:]).mean())

        elif self.strategy == 'macd':
            time_window = self.time_window // 2
            time_window_short = time_window // 2
            macd, macdsignal, macdhist = talib.MACD(price_list, fastperiod = time_window_short, slowperiod = time_window, signalperiod = time_window_short - 3)
            return float(macdhist[-3:].mean())

class MeanRevertAgent(Agent):
    num_of_agent = 0
    def __init__(self, _id, start_cash: int = 1000000, start_securities: Dict[str, int] = None, risk_preference = 0.5, strategy = None, time_window = None):
        super().__init__(_id, 'mr', start_cash, start_securities,risk_preference)
        MeanRevertAgent.add_counter()
        self.strategy = strategy
        self.time_window = time_window
        self.trading_probability = max(0.02 * risk_preference, 0.01)
    
    def step(self):
        super().step()
        if np.random.binomial(n = 1, p = self.trading_probability) == 1:
            self.generate_order()

    def generate_order(self):
        code = "TSMC"
        stock_size = self.core.get_stock_size()
        tick_size = self.core.get_tick_size(code)
        current_price = self.core.get_current_price(code)
        price_list = self.core.get_records(code = code, _type = 'close', step = self.time_window)

        if len(price_list) < self.time_window:
            return
        if self.strategy == 'ema':
            self.strategy
        risk_free_amount = (1-self.risk_preference) * self.wealth
        signal = self.get_signal(price_list)
        # add hard limit to sell
        if self.holdings[code] > 0:
            profit_ratio = (current_price / self.average_cost)
            # realize
            if current_price > self.average_cost:
                realize_probability = profit_ratio * (1-self.risk_preference)
                if realize_probability > 1 or np.random.binomial(n = 1, p = realize_probability) == 1:
                    self.place_limit_ask_order(code, quantity = self.holdings[code], price = current_price)
            # flog
            elif profit_ratio < 0.9:
                # the range of loss is -10% ~ -50%, depend on the risk preference
                flog_probability = (1 - profit_ratio) * 2 * (1 - self.risk_preference)
                if flog_probability > 1 or np.random.binomial(n = 1, p = flog_probability) == 1:
                    self.place_limit_ask_order(code, quantity = self.holdings[code], price = current_price)

        if signal < 0 and self.cash > risk_free_amount:
            best_bid = self.core.get_best_bids('TSMC', 1)
            price = round(current_price + np.random.randint(-5, 1) * tick_size , 2)
            bid_quantity = round( (self.cash - risk_free_amount) / (stock_size*price))
            self.place_limit_bid_order(code, quantity = bid_quantity, price = price)

        elif signal > 0 and self.cash < risk_free_amount and self.holdings[code] > 0:
            best_ask = self.core.get_best_asks('TSMC', 1)
            price = round(current_price + np.random.randint(-1, 5) * tick_size , 2)
            ask_quantity = min(round( (risk_free_amount - self.cash) / (stock_size*price)), self.holdings[code])
            self.place_limit_ask_order(code, quantity = ask_quantity, price = price)

    def get_signal(self, price_list, volumn_list = None):
        price_list = np.array(price_list, dtype='float64')
        if self.strategy == 'sma':
            time_window_short = self.time_window // 2
            long = talib.SMA(price_list, self.time_window - 3)
            short = talib.SMA(price_list, time_window_short - 3)
            return float( (short[-3:] - long[-3:]).mean())

        elif self.strategy == 'ema':
            time_window_short = self.time_window // 2
            long = talib.EMA(price_list, self.time_window - 3)
            short = talib.EMA(price_list, time_window_short - 3)
            return float( (short[-3:] - long[-3:]).mean())

        elif self.strategy == 'macd':
            time_window = self.time_window // 2
            time_window_short = time_window // 2
            macd, macdsignal, macdhist = talib.MACD(price_list, fastperiod = time_window_short, slowperiod = time_window, signalperiod = time_window_short - 3)
            return float(macdhist[-3:].mean())


class MomentumAgent(Agent):
    num_of_agent = 0
    def __init__(self, _id, start_cash: int = 1000000, start_securities: Dict[str, int] = None, time_window = 100):
        super().__init__(_id, 'mt', start_cash, start_securities)
        MomentumAgent.add_counter()
        self.time_window = time_window
        self.local_minimum = None
        self.local_maximum = None

    def step(self):
        super().step()
        if np.random.binomial(n = 1, p = 0.02) == 1:
            self.generate_order()

    def generate_order(self):
        code = "TSMC"
        price_list = self.core.get_records(code = code, _type = 'close', step = self.time_window)
        if self.local_minimum == None:    
            self.local_minimum = min(price_list)
        if self.local_maximum == None:
            self.local_maximum = max(price_list)
        
        current_price = self.core.get_current_price(code)
        trade_propotion = np.random.random()
        # trend is negative(bid signal) and need to bid
        # if current_price < value:
        #     bid_or_ask = 'BID'
        #     quantity = math.floor((trade_propotion * self.cash) / (100*price))
        #     self.place_limit_bid_order(code, quantity, value)
        # elif current_price > value:
        #     bid_or_ask = 'ASK'
        #     quantity = math.floor(trade_propotion * self.holdings[code])
        #     self.place_limit_ask_order(code, quantity, value)
    
class FundamentalistAgent(Agent):
    num_of_agent = 0
    def __init__(self, _id, start_cash: int = 1000000, start_securities: Dict[str, int] = None, securities_value = None):
        super().__init__(_id, 'mr', start_cash, start_securities)
        FundamentalistAgent.add_counter()
        self.securities_value = securities_value
    
    def step(self):
        super().step()
        if np.random.binomial(n = 1, p = 0.02) == 1:
            self.generate_order()

    def generate_order(self):
        code = "TSMC"
        stock_size = self.core.get_stock_size()
        value = self.securities_value[code]
        price = value
        current_price = self.core.get_current_price(code)

        price_ratio = current_price/value
        if price_ratio < 1:
            trade_propotion = 1 - price_ratio
        elif price_ratio > 1:
            trade_propotion = price_ratio - 1

        if current_price < value:
            bid_or_ask = 'BID'
            quantity = math.floor((trade_propotion * self.cash) / (stock_size*price))
            self.place_limit_bid_order(code, quantity, value)
        elif current_price > value:
            bid_or_ask = 'ASK'
            quantity = math.floor(trade_propotion * self.holdings[code])
            self.place_limit_ask_order(code, quantity, value)

class DahooAgent(Agent):
    num_of_agent = 0
    def __init__(self, _id, start_cash: int = 1000000, start_securities: Dict[str, int] = None, securities_value = None):
        super().__init__(_id, 'mr', start_cash, start_securities)
        DahooAgent.add_counter()
        self.securities_value = securities_value
        self.status = 'HOLD'
        self.strategy_record = []
        self.cool_down = 0

    def step(self):
        super().step()
        if self.cool_down > 0:
            self.cool_down -= 1
            return

        if self.status == 'HOLD' and (self.get_time() == 100 or (self.get_time()+1) % 1000 == 0):
            self.plan()
            self.status = 'COLLECT'
        elif self.status == 'SUPRESS':
            self.supress_price()
        elif self.status == 'COLLECT':
            self.collect_chip()
        elif self.status == 'RAISE':
            self.raise_price()
        elif self.status == 'DUMP':
            self.dump_security()
        elif self.status == 'SUM':
            if len(self.orders) > 0:
                return
            self.strategy_record[-1]['final_wealth'] = self.wealth
            self.status = 'HOLD'
    
    def plan(self):
        start_time = self.get_time()
        start_wealth = self.wealth
        value = self.core.get_value('TSMC')
        supress_price = value * 0.95
        collect_price = value * 0.95
        raise_price = value * 1.05
        dump_price = value * 0.85
        strategy = {
            'start_time': start_time,
            'start_wealth': start_wealth,
            'supress_price': supress_price,
            'collect_price': collect_price,
            'raise_price': raise_price,
            'dump_price': dump_price
        }
        self.strategy_record.append(strategy)

    def supress_price(self):
        code = "TSMC"
        value = self.securities_value[code]
        current_price = self.core.get_current_price(code)
        supress_price = self.strategy_record[-1]['supress_price']
        if current_price < supress_price:
            self.status = 'COLLECT'
        else:
            self.place_limit_bid_order(code, self.holdings // 5, 80)
        
    def collect_chip(self):
        # hold large number of stocks and try to snatch the chives
        # collect chip
        code = "TSMC"
        stock_size = self.core.get_stock_size()
        current_price = self.core.get_current_price(code)
        value = self.securities_value[code]
        collect_price = self.strategy_record[-1]['collect_price']
        if 'collect_number' not in self.strategy_record[-1].keys():
            collect_propotion = self.strategy_record[-1]['start_wealth'] * 0.6
            self.strategy_record[-1]['collect_number'] = collect_propotion // (stock_size*collect_price)

        if current_price < collect_price:
            tick_size = self.core.get_tick_size(code)
            price = current_price - np.random.randint(1, 5) * tick_size
            quantity = 100
            self.place_limit_bid_order(code, quantity, price)
            self.strategy_record[-1]['collect_number'] -= quantity
            if self.strategy_record[-1]['collect_number'] < 0:
                self.strategy_record[-1].pop('collect_number')
                self.status = 'RAISE'
    
    def raise_price(self):
        # hold large number of stocks and try to snatch the chives
        code = "TSMC"
        stock_size = self.core.get_stock_size()
        current_price = self.core.get_current_price(code)
        value = self.securities_value[code]
        tick_size = self.core.get_tick_size(code)
        raise_price = self.strategy_record[-1]['raise_price']
        if 'raise_number' not in self.strategy_record[-1].keys():
            raise_propotion = self.strategy_record[-1]['start_wealth'] * 0.35
            self.strategy_record[-1]['raise_times'] = 10
            self.strategy_record[-1]['raise_number'] = (raise_propotion // (stock_size*raise_price)) // self.strategy_record[-1]['raise_times']
            self.strategy_record[-1]['raise_tick'] = ((raise_price - current_price) // tick_size) // self.strategy_record[-1]['raise_times']

        price = current_price + self.strategy_record[-1]['raise_tick'] * tick_size
        quantity = self.strategy_record[-1]['raise_number']
        self.place_limit_bid_order(code, quantity, price)
        self.strategy_record[-1]['raise_times'] -= 1
        self.cool_down = 10
        if self.strategy_record[-1]['raise_times'] == 0:
            self.status = 'DUMP'
            self.strategy_record[-1].pop('raise_number')
            self.strategy_record[-1].pop('raise_times')
            self.strategy_record[-1].pop('raise_tick')
    
    def dump_security(self):
        # hold large number of stocks and try to snatch the chives
        code = "TSMC"
        stock_size = self.core.get_stock_size()
        current_price = self.core.get_current_price(code)
        tick_size = self.core.get_tick_size(code)
        dump_price = self.strategy_record[-1]['dump_price']
        if 'dump_number' not in self.strategy_record[-1].keys():
            self.strategy_record[-1]['dump_times'] = 3
            self.strategy_record[-1]['dump_number'] = self.holdings[code] // self.strategy_record[-1]['dump_times']
            self.strategy_record[-1]['dump_tick'] = ((dump_price - current_price) // tick_size) // self.strategy_record[-1]['dump_times']

        price = current_price + self.strategy_record[-1]['dump_tick'] * tick_size
        quantity = self.strategy_record[-1]['dump_number']
        self.place_limit_bid_order(code, quantity, price)
        self.strategy_record[-1]['dump_times'] -= 1
        self.cool_down = 10
        if self.strategy_record[-1]['dump_times'] == 0:
            self.status = 'SUM'
            self.strategy_record[-1].pop('dump_number')
            self.strategy_record[-1].pop('dump_times')
            self.strategy_record[-1].pop('dump_tick')


class BrokerAgent(Agent):
    num_of_agent = 0
    def __init__(self, _id, code, start_cash = 200000000, target_volume = 500):
        super().__init__(_id, 'BROKER', start_cash)
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


class TestAgent(Agent):
    num_of_agent = 0
    
    def __init__(self, _id, order_list, start_cash = 1000000, start_securities = None):
        super().__init__(_id, 'te', start_cash, start_securities)
        TestAgent.add_counter()
        self.order_list = order_list

    def step(self):
        super().step()
        # to trade?
        if len(self.order_list) != 0 and self.order_list[0]['time'] == self.get_time():
            order = self.order_list.pop(0)
            if order['bid_or_ask'] == 'BID':
                self.place_limit_bid_order(order['code'], order['quantity'], order['price'])
            else:
                self.place_limit_ask_order(order['code'], order['quantity'], order['price'])