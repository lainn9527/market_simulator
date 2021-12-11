import math
import numpy as np
import random
# import talib
from numpy.lib.function_base import quantile
from datetime import datetime, timedelta, time
from typing import DefaultDict, Dict, List
from collections import defaultdict

from core.message import Message
from core.order import LimitOrder, MarketOrder, ModificationOrder

class Agent:
    num_of_agent = 0

    def __init__(self, _id, _type, start_cash = 10000000, start_securities = None, average_cost = 100, risk_preference = 1):
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
        self.average_cost = average_cost
        self.risk_preference = risk_preference

        self.reserved_cash = 0
        self.reserved_holdings = {code: 0 for code in self.holdings.keys()}
        self.wealth = 0
        self.orders = {code: [] for code in self.holdings.keys()}
        self.orders_history = {code: [] for code in self.holdings.keys()}
        self.cancelled_orders = {code: [] for code in self.holdings.keys()}
        self.modified_orders = {code: [] for code in self.holdings.keys()}
        self.initial_wealth = 0
        self.bids_volume = 0
        self.asks_volume = 0

        # state flag
        
    
    def start(self, core):
        self.core = core
        self.initial_wealth = self.cash + sum([self.core.get_value(code) * num * self.core.get_stock_size() for code, num in self.holdings.items()])
        # self.average_cost = self.core.get_value('TSMC')
        self.wealth = self.initial_wealth
        self.average_cost = round(random.gauss(mu = self.core.get_value('TSMC'), sigma = 10), 1)

    def step(self):
        self.update_wealth()


    def receive_message(self, message):
        if message.receiver != self.unique_id and message.receiver != 'agents':
            raise Exception('Wrong receiver!')

        message_subject = ['OPEN_SESSION', 'CLOSE_SESSION', 'OPEN_AUCTION', 'CLOSE_AUCTION', 'OPEN_CONTINUOUS_TRADING', 'STOP_CONTINUOUS_TRADING', 'ORDER_PLACED', 'ORDER_CANCELLED', 'ORDER_INVALIDED', 'ORDER_FILLED', 'ORDER_FINISHED', 'ISSUE_INTEREST', 'ISSUE_DIVIDEND']

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

        elif message.subject == 'ORDER_CANCEL_FAILED':
            # remove from the cancelled list
            if message.content['order_id'] in self.cancelled_orders[message.content['code']]:
                self.cancelled_orders[message.content['code']].remove(message.content['order_id'])


        elif message.subject == 'ISSUE_INTEREST':
            interest_rate = message.content['interest_rate']
            self.cash += round(self.cash * interest_rate, 2)

        elif message.subject == 'ISSUE_DIVIDEND':
            for code, number in self.holdings.items():
                dividend = self.core.get_stock_size() * message.content[code] * number
                self.cash += round(dividend, 2)

        elif message.subject == 'ORDER_MODIFIED':
            pass
            # if message.content['original_order_id'] in self.orders[message.content['code']] or message.content['new_order_id'] not in self.orders[message.content['code']]:
            #     raise Exception
            # self.orders[message.content['code']].append(message.content['new_order_id'])

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

        # Check if there are ask orders and if so, hedge them.
        # self.check_bid_hedge(code, price, quantity)

        if price <= 0 or quantity == 0 or cost > self.cash:
            return
                
        self.cash -= cost
        self.reserved_cash += cost
        self.bids_volume += quantity

        order = LimitOrder(self.unique_id, code, 'LIMIT', 'BID', quantity, price)
        msg = Message('MARKET', 'LIMIT_ORDER', self.unique_id, 'market', {'order': order})
        
        self.core.send_message(msg)
    
    def place_limit_ask_order(self, code, quantity, price):
        tick_size = self.core.get_tick_size(code)
        price = round(tick_size * ((price+1e-6) // tick_size), 2)

        # Check if there are bid orders and if so, hedge them.
        # self.check_ask_hedge(code, price, quantity)

        
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
    
    def cancel_order(self, code, order_id):
        self.cancelled_orders[code].append(order_id)
        msg = Message('MARKET', 'CANCELLATION_ORDER', self.unique_id, 'market', {'code': code, 'order_id': order_id})
        self.core.send_message(msg)

    def modify_order(self, code, order_id, price, volume):
        order_record = self.core.get_order_record(code, order_id)
        original_price = order_record.order.price
        original_quantity = order_record.order.quantity
        filled_quantity = order_record.filled_quantity
        if volume <= filled_quantity:
            # the modified volume <= filled volume which is error
            raise Exception
        
        if order_record.order.bid_or_ask == 'BID':
            # check validility
            tick_size = self.core.get_tick_size(code)
            stock_size = self.core.get_stock_size()
            price = round(tick_size * ((price+1e-6) // tick_size), 2)
            transaction_rate = self.core.get_transaction_rate()
            reserved_cost = round((original_quantity - filled_quantity) * original_price * stock_size * (1 + transaction_rate), 2) 
            cost = round((volume - filled_quantity) * price * stock_size * (1 + transaction_rate), 2)
            supply_cost = round(cost - reserved_cost, 2)
            if supply_cost > self.cash:
                # this modification is not available
                return
            elif supply_cost > 0:
                # deduct here, refund in message
                self.cash -= supply_cost
                self.reserved_cash += supply_cost
            self.bids_volume += volume - original_quantity

        elif order_record.order.bid_or_ask == 'ASK':
            supply_quantity = volume - original_quantity
            if supply_quantity > self.holdings[code]:
                # this modification is not available
                return
            elif supply_quantity > 0:
                # deduct here, refund in message
                self.holdings[code] -= supply_quantity
                self.reserved_holdings[code] += supply_quantity
            
            self.asks_volume += volume - original_quantity

        modification_order = ModificationOrder.from_limit_order(limit_order = order_record.order, price = price, quantity = volume, order_id = order_id)
        msg = Message('MARKET', 'MODIFICATION_ORDER', self.unique_id, 'market', {'order': modification_order})
        self.core.send_message(msg)
    
    def check_bid_hedge(self, code, price, quantity):
        for order in self.orders[code]:
            if order in self.cancelled_orders[code]:
                continue
            order_record = self.core.get_order_record(code, order)
            if order_record.order.bid_or_ask == 'ASK' and price >= order_record.order.price:
                # cancel all paradoxical order
                self.cancel_order(code, order_record.order.order_id)


    def check_ask_hedge(self, code, price, quantity):
        for order in self.orders[code]:
            if order in self.cancelled_orders[code]:
                continue
            order_record = self.core.get_order_record(code, order)
            if order_record.order.bid_or_ask == 'BID' and price <= order_record.order.price:
                # cancel all paradoxical order
                self.cancel_order(code, order_record.order.order_id)

    def handle_filled_order(self, code, order_id, price, quantity, transaction_cost):
        stock_size = self.core.get_stock_size()
        order_record = self.core.get_order_record(code, order_id)
        transaction_rate = self.core.get_transaction_rate()
        if order_record.order.bid_or_ask == 'BID':
            self.average_cost = round( (stock_size*(self.average_cost * self.holdings[code] + price * quantity) + transaction_cost) / (stock_size*(self.holdings[code] + quantity)), 2)
            self.holdings[code] += quantity
            self.cash += (order_record.order.price - price) * quantity * stock_size * (1 + transaction_rate)
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
        
        if order_id in self.cancelled_orders[code]:
            self.cancelled_orders[code].remove(order_id)
        self.orders[code].remove(order_id)
        self.orders_history[code].append(order_id)

    def update_wealth(self):
        cash = self.cash + self.reserved_cash
        securities_value = 0
        for code in self.holdings.keys():
            price = self.core.get_current_price(code)
            securities_value +=  price * (self.holdings[code] + self.reserved_holdings[code]) * self.core.get_stock_size() 
        self.wealth = cash + securities_value

    def get_time(self):
        return self.core.timestep

    def log_event(self, event_type, event):
        # print(f"Agent: {self.unique_id}, Event type: {event_type}, Event: {event}")
        return
    
    def handle_open(self):
        return

    
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
    def __init__(self, _id, start_cash: int = 1000000, start_securities: Dict[str, int] = None, average_cost = 100, risk_preference = 1):
        super().__init__(_id = _id, 
                         _type = 'rl',
                         start_cash = start_cash,
                         start_securities = start_securities,
                         average_cost = average_cost,
                         risk_preference = 1)

        RLAgent.add_counter()
        self.action_status = RLAgent.HOLD
    
    def step(self, action = None):
        super().step()
        if isinstance(action, type(None)):
            return
        elif isinstance(action, np.ndarray):
            action.tolist()


        bid_or_ask = action[0]
        ticks = action[1]
        volume = action[2]
        # ticks = random.randint(1, 10)
        # volume = random.randint(1, 5)
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
            price = round(current_price + ticks * tick_size, 1)
            self.place_limit_bid_order('TSMC', float(volume), float(price))

        elif bid_or_ask == 1:
            # ask
            self.action_status = RLAgent.INVALID_ACTION
            best_ask = self.core.get_best_asks('TSMC', 1)
            best_ask = current_price if len(best_ask) == 0 else best_ask[0]['price']
            price = round(current_price - ticks * tick_size, 1)
            self.place_limit_ask_order('TSMC', float(volume), float(price))

    def receive_message(self, message):
        super().receive_message(message)
        if message.subject == 'ORDER_PLACED':
            self.action_status = RLAgent.VALID_ACTION

class ScalingAgent(Agent):
    num_of_agent = 0
    num_opt_noise_agent = 0
    num_pes_noise_agent = 0
    num_fundamentalist_agent = 0

    def __init__(self,
                 _id,
                 start_cash = 1000000,
                 start_securities = None,
                 average_cost = 100,
                 range_of_price = 5,
                 range_of_quantity = 5,
                 risk_preference = 1,
                 group = 'optimistic'):
                
        super().__init__(_id = _id, 
                         _type = 'sc',
                         start_cash = start_cash,
                         start_securities = start_securities,
                         average_cost = average_cost,
                         risk_preference = risk_preference)
                         
        ScalingAgent.add_counter()
        self.range_of_quantity = range_of_quantity
        self.range_of_price = range_of_price
        self.group = group # [optimistic, pessimistic, fundamentalist]
        self.time_delta = 0
        self.v_1 = 10
        self.v_2 = 0.6
        self.beta = 4
        self.alpha_1 = 0.6
        self.alpha_2 = 1.5
        self.alpha_3 = 1
        self.fundamentalist_discount = 0.75
        self.precision = 100
        # self.return_rate_range = 20
        self.return_rate_range = np.random.randint(10, 100)
        
        if self.group == 'optimistic':
            ScalingAgent.num_opt_noise_agent += 1

        elif self.group == 'pessimistic':
            ScalingAgent.num_pes_noise_agent += 1

        elif self.group == 'fundamentalist':
            ScalingAgent.num_fundamentalist_agent += 1
        
    def step(self):
        super().step()
        self.check_group()
        self.generate_order()

    def check_group(self):
        market_stats = self.core.get_call_env_state(lookback = self.return_rate_range, from_last = False)
        current_value = self.core.get_current_value('TSMC')
        # current_value = self.core.get_current_value_with_noise('TSMC')
        current_price = market_stats['price'][-1]
        d_p = np.diff(market_stats['price']).mean().item()
        return_rate = d_p / current_price

        # check opinion
        if self.group == 'optimistic' and self.optimistic_to_pessimistic(return_rate, current_price):
            self.group = 'pessimistic'
            ScalingAgent.num_opt_noise_agent -= 1
            ScalingAgent.num_pes_noise_agent += 1
            
        elif self.group == 'pessimistic' and self.pessimistic_to_optimistic(return_rate, current_price):
            self.group = 'optimistic'
            ScalingAgent.num_pes_noise_agent -= 1
            ScalingAgent.num_opt_noise_agent += 1

        # check group
        self.switch_noist_fundamentalist(d_p, current_price, current_value)


    def pessimistic_to_optimistic(self, return_rate, price):
        num_noise_agent = ScalingAgent.num_opt_noise_agent + ScalingAgent.num_pes_noise_agent
        opinion_prop = (ScalingAgent.num_opt_noise_agent - ScalingAgent.num_pes_noise_agent) / num_noise_agent
        utility = self.alpha_1 * opinion_prop + (self.alpha_2/self.v_1) * return_rate/price
        prob = self.v_1 * (num_noise_agent / ScalingAgent.num_of_agent) * math.exp(utility) / self.precision
        if prob > 1 or np.random.binomial(n = 1, p = prob) == 1:
            return True
        else:
            return False

    def optimistic_to_pessimistic(self, return_rate, price):
        num_noise_agent = ScalingAgent.num_opt_noise_agent + ScalingAgent.num_pes_noise_agent
        opinion_prop = (ScalingAgent.num_opt_noise_agent - ScalingAgent.num_pes_noise_agent) / num_noise_agent
        utility = self.alpha_1 * opinion_prop + (self.alpha_2/self.v_1) * return_rate/price
        prob = self.v_1 * (num_noise_agent / ScalingAgent.num_of_agent) * math.exp(-utility) / self.precision

        if prob > 1 or np.random.binomial(n = 1, p = prob) == 1:
            return True
        else:
            return False

    def switch_noist_fundamentalist(self, d_p, price, value):
        original_group = self.group
        risk_free_rate = self.core.get_risk_free_rate()
        dividends = risk_free_rate * value
        # dividends = 0

        chartist_profit = (dividends + (1 / self.v_2)*d_p) / price - risk_free_rate
        fundamentalist_profit = self.fundamentalist_discount * abs( (value - price) / price)
        utility_21 = self.alpha_3 * (chartist_profit - fundamentalist_profit)
        utility_22 = self.alpha_3 * (-chartist_profit - fundamentalist_profit)

        if original_group == 'fundamentalist':
            fund_to_opt_prob = self.v_2 * (ScalingAgent.num_opt_noise_agent / ScalingAgent.num_of_agent) * math.exp(utility_21) / self.precision
            fund_to_pes_prob = self.v_2 * (ScalingAgent.num_pes_noise_agent / ScalingAgent.num_of_agent) * math.exp(utility_22) / self.precision
            if fund_to_opt_prob >= fund_to_pes_prob and (fund_to_opt_prob > 1 or np.random.binomial(n = 1, p = fund_to_opt_prob) == 1):
                self.group = 'optimistic'
                ScalingAgent.num_fundamentalist_agent -=1
                ScalingAgent.num_opt_noise_agent += 1
                
            elif fund_to_pes_prob > 1 or np.random.binomial(n = 1, p = fund_to_pes_prob) == 1:
                self.group = 'pessimistic'
                ScalingAgent.num_fundamentalist_agent -=1
                ScalingAgent.num_pes_noise_agent += 1


        elif original_group == 'optimistic':
            opt_to_fund_prob = self.v_2 * (ScalingAgent.num_fundamentalist_agent / ScalingAgent.num_of_agent) * math.exp(-utility_21) / self.precision
            if opt_to_fund_prob > 1 or np.random.binomial(n = 1, p = opt_to_fund_prob) == 1:
                self.group = 'fundamentalist'
                ScalingAgent.num_opt_noise_agent -= 1
                ScalingAgent.num_fundamentalist_agent +=1

        elif original_group == 'pessimistic':
            pes_to_fund_prob = self.v_2 * (ScalingAgent.num_fundamentalist_agent / ScalingAgent.num_of_agent) * math.exp(-utility_22) / self.precision
            if pes_to_fund_prob > 1 or np.random.binomial(n = 1, p = pes_to_fund_prob) == 1:
                self.group = 'fundamentalist'
                ScalingAgent.num_pes_noise_agent -= 1
                ScalingAgent.num_fundamentalist_agent +=1

    def generate_order(self):
        # which one?
        code = np.random.choice(list(self.holdings.keys()))
        market_stats = self.core.get_call_env_state(lookback = self.return_rate_range, from_last = False)
        d_p = np.diff(market_stats['price']).mean().item()

        current_price = self.core.get_current_price(code)
        current_value = self.core.get_current_value(code)
        # current_value = self.core.get_current_value_with_noise(code)
        # bid = market_stats['bid']
        # ask = market_stats['ask']


        last_filled_bid = current_price
        last_filled_ask = current_price
        tick_size = self.core.get_tick_size(code)
        stock_size = self.core.get_stock_size()

        previous_price = market_stats['price'][-2]

        # quantity = np.random.randint(1, self.range_of_quantity)
        if self.group == 'optimistic':
            tick_range = d_p // tick_size
            premium_ticks =  np.random.randint(-self.range_of_price + tick_range, self.range_of_price + tick_range)
            bid_price = round(current_price + premium_ticks * tick_size , 1)
            available_bid_quantity = self.cash // (bid_price * stock_size)
            bid_quantity = max(round(np.random.random() * self.range_of_quantity * available_bid_quantity), 1)
            # bid_quantity = np.random.randint(1, 5)
            self.place_limit_bid_order(code, bid_quantity, bid_price)

        elif self.group == 'pessimistic':
            tick_range =  d_p // tick_size
            premium_ticks = np.random.randint(-self.range_of_price + tick_range, self.range_of_price + tick_range) 
            ask_price = round(current_price + premium_ticks * tick_size , 1)
            available_ask_quantity = self.holdings[code]
            ask_quantity = max(round(np.random.random() * self.range_of_quantity * available_ask_quantity),  1)
            # ask_quantity = np.random.randint(1, 5)
            self.place_limit_ask_order(code, min(ask_quantity, self.holdings[code]), ask_price)

        elif self.group == 'fundamentalist':
            diff = current_value - current_price
            tick_range = diff // tick_size
            level = abs(diff / current_value)
            if diff > 0:
                premium_ticks = np.random.randint(-self.range_of_price + tick_range, tick_range + self.range_of_price)
                bid_price = round(current_price + premium_ticks * tick_size , 1)
                available_bid_quantity = self.cash // (bid_price * stock_size)
                # bid_quantity = max(round(np.random.random() * self.range_of_quantity * available_bid_quantity), 1)
                # bid_quantity = np.random.randint(1, 5)
                bid_quantity = max(round(level * available_bid_quantity), 1)
                self.place_limit_bid_order(code, bid_quantity, bid_price)

            elif diff < 0:
                premium_ticks = np.random.randint(-self.range_of_price + tick_range, self.range_of_price + tick_range)
                ask_price = round(current_price + premium_ticks * tick_size , 1)
                available_ask_quantity = self.holdings[code]
                # ask_quantity = max(round(np.random.random() * self.range_of_quantity * available_ask_quantity) , 1)
                # ask_quantity = np.random.randint(1, 5)
                ask_quantity = max(round(level * available_ask_quantity) , 1)
                self.place_limit_ask_order(code, ask_quantity, ask_price)
                
        if current_price == previous_price:
            a = 100
        if current_price - current_value > 3:
            a = 100

    
    @classmethod
    def get_opt_number(cls):
        return ScalingAgent.num_opt_noise_agent
    
    @classmethod
    def get_pes_number(cls):
        return ScalingAgent.num_pes_noise_agent

    @classmethod
    def get_fud_number(cls):
        return ScalingAgent.num_fundamentalist_agent

    @classmethod
    def reset_agent_number(cls):
        ScalingAgent.num_opt_noise_agent = 0
        ScalingAgent.num_pes_noise_agent = 0
        ScalingAgent.num_fundamentalist_agent = 0



class ZeroIntelligenceAgent(Agent):
    num_of_agent = 0
    
    def __init__(self, _id, start_cash = 1000000, start_securities = None, average_cost = 100, risk_preference = 1, bid_side = 0.5, range_of_price = 5, range_of_quantity = 5):
        super().__init__(_id = _id, 
                         _type = 'zi',
                         start_cash = start_cash,
                         start_securities = start_securities,
                         average_cost = average_cost,
                         risk_preference = risk_preference)
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




class TestAgent(Agent):
    num_of_agent = 0
    
    def __init__(self, _id, order_list, start_cash = 1000000, start_securities = None, average_cost = 100):
        super().__init__(_id, 'te', start_cash, start_securities, average_cost)
        TestAgent.add_counter()
        self.order_list = order_list

    def step(self):
        super().step()
        # to trade?
        i = 0
        while i < len(self.order_list):
            if self.order_list[i]['time'] == self.get_time():
                order = self.order_list.pop(i)
                if order['bid_or_ask'] == 'BID':
                    self.place_limit_bid_order(order['code'], order['quantity'], order['price'])
                else:
                    self.place_limit_ask_order(order['code'], order['quantity'], order['price'])
            else:
                i += 1
