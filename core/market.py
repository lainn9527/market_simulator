import random
import math
import numpy as np
from .order_book import OrderBook, CallOrderBook
from .message import Message
from .order import LimitOrder, MarketOrder

class Market:
    def __init__(self, core, interest_rate, interest_period, securities, clear_period, transaction_rate):
        self.core = core
        self.is_trading = False
        self.stock_size = 100
        self.interest_rate = interest_rate
        self.interest_period = interest_period
        self.clear_period = clear_period
        self.transaction_rate = transaction_rate
        self.orderbooks = {code: OrderBook(self, code, **value) for code, value in securities.items()}
    
    def step(self):
        timestep = self.get_time()
        for orderbook in self.orderbooks.values():
            orderbook.step_summarize()

        # self.change_value()

        if timestep > 0:
            if timestep % self.clear_period == 0:
                for orderbook in self.orderbooks.values():
                    orderbook.clear_orders()

            if timestep % self.interest_period == 0:
                self.issue_interest()
        
        
    def open_session(self):
        # determine the price list of orderbook
        for orderbook in self.orderbooks.values():
            orderbook.set_price()

        # notify agents
        msg = Message('ALL_AGENTS', 'OPEN_SESSION', 'market', 'agents', None)
        self.announce_message(msg)
        self.is_trading = True
        

    def close_session(self):
        '''
        Works to be done at the end of a market day:
        1. clear the order and notify the agents
        2. summarize trading information today
        3. send the daily stats to agents
        '''

        daily_info = {}
        close_price = {}
        for code, orderbook in self.orderbooks.items():
            # get the daily info
            daily_info[code] = orderbook.daily_summarize()

        self.announce_message(Message('ALL_AGENTS', 'CLOSE_SESSION', 'market', 'agents', daily_info), self.get_time())

    def send_message(self, message):
        self.core.send_message(message)

    def announce_message(self, message):
        self.core.announce_message(message)

    def receive_message(self, message):
        if message.receiver != 'market':
            raise Exception
        
        if message.subject == 'LIMIT_ORDER':
            if not isinstance(message.content['order'], LimitOrder):
                raise Exception
            self.orderbooks[message.content['order'].code].handle_limit_order(message.content['order'])
            

        elif message.subject == 'MARKET_ORDER':
            if not isinstance(message.content['order'], MarketOrder):
                raise Exception
            self.orderbooks[message.content['order'].code].handle_market_order(message.content['order'])
        
        elif message.subject == 'CANCELLATION_ORDER':
            self.orderbooks[message.content['code']].cancel_order(message.content['order_id'])
        
        elif message.subject == 'MODIFICATION_ORDER':
            self.orderbooks[message.content['order'].code].modify_order(message.content['order'])

        else:
            raise Exception

    def issue_interest(self):
        self.send_message(Message('ALL_AGENTS', 'ISSUE_INTEREST', 'MARKET', 'agents', {'interest_rate': self.interest_rate}))

    def issue_dividends(self):
        for code, orderbook in self.orderbooks.items():
            current_value = self.get_value(code)
            dividend = current_value * self.interest_rate
            self.send_message(Message('ALL_AGENTS', 'ISSUE_DIVIDEND', 'MARKET', 'agents', {code: dividend}))

            # previous = list(orderbook.dividend_record.values)[-1] if len(orderbook.dividend_record) > 0 else orderbook.value
            # dividend = orderbook.value + orderbook.dividend_ar * (previous - orderbook.value) + random.gauss(0, orderbook.dividend_var)
            # for order_record in orderbook.current_orders:
            #     self.send_message(Message('AGENT', 'ISSUE_DIVIDEND', 'MARKET', order_record.order.orderer, {'code': code, 'dividend': dividend}))
            orderbook.dividend_record[self.get_time()] = dividend
        
    def change_value(self, code):
        v = random.gauss(0, 0.005)
        previous_value = self.get_value(code)
        current_value = round(math.exp(math.log(previous_value) + v), 1)
        self.orderbooks[code].adjust_value(current_value)

    def random_event(self, code):
        v = random.gauss(0, 0.05)
        previous_value = self.get_value(code)
        current_value = round(math.exp(math.log(previous_value) + v), 1)
        self.orderbooks[code].adjust_value(current_value)


    def check_volatility(self):
        for code, orderbook in self.orderbooks.items():
            if len(orderbook.steps_record['average']) < 2:
                return
            diff = abs(orderbook.steps_record['average'][-1] - orderbook.steps_record['average'][-2])
            volatility = diff / orderbook.steps_record['average'][-2]
            if volatility > 0.1:
                print('high vol warning')

    def check_liquidity(self):
        if self.get_time() < 100:
            return
        for code, orderbook in self.orderbooks.items():
            if len(orderbook.bids_price) < 3 or len(orderbook.asks_price) < 3:
                print('Liquidity error')
        
    def get_order_record(self, code, order_id):
        return self.orderbooks[code].orders[order_id]

    def get_current_price(self, code):
        return self.orderbooks[code].current_record['price']
    
    def get_records(self, code, _type, step = 1, from_last = True):
        if _type in ['close', 'average', 'open'] and _type not in self.orderbooks[code].steps_record.keys():
            _type = 'price'
        if from_last:
            return self.orderbooks[code].steps_record[_type][:-1*(step+1):-1]
        else:
            return self.orderbooks[code].steps_record[_type][-1*step::]

    def get_best_bids(self, code, number = 1):
        return [{'price': price, 'volume': self.orderbooks[code].bids_volume[price]} for price in self.orderbooks[code].bids_price[:number]]

    def get_best_asks(self, code, number = 1):
        return [{'price': price, 'volume': self.orderbooks[code].asks_volume[price]} for price in self.orderbooks[code].asks_price[:number]]
    
    def get_best_bid_price(self, code, number = 1):
        return self.orderbooks[code].bids_price[:number]

    def get_best_ask_price(self, code, number = 1):
        return self.orderbooks[code].asks_price[:number]

    def get_average_bid(self, code):
        if len(self.orderbooks[code].bids_price) == 0:
            return self.get_current_price(code)
        else:
            prices = np.array(list(self.orderbooks[code].bids_volume.keys()))
            volumes = np.array(list(self.orderbooks[code].bids_volume.values()))
            return round((np.dot(prices, volumes) / volumes.sum()).item(), 1)

    def get_average_ask(self, code):
        if len(self.orderbooks[code].asks_price) == 0:
            return self.get_current_price(code)
        else:
            prices = np.array(list(self.orderbooks[code].asks_volume.keys()))
            volumes = np.array(list(self.orderbooks[code].asks_volume.values()))
            return round((np.dot(prices, volumes) / volumes.sum()).item(), 1)

    def get_tick_size(self, code):
        return self.orderbooks[code].tick_size

    def get_stock_size(self):
        return self.stock_size

    def get_transaction_rate(self):
        return self.transaction_rate

    def get_risk_free_rate(self):
        return self.interest_rate

    def get_securities(self):
        return list(self.orderbooks.keys())

    def get_time(self):
        return self.core.timestep
    
    def get_value(self, code):
        if type(self.orderbooks[code].value) is list:
            return self.orderbooks[code].value[-1]
        else:
            return self.orderbooks[code].value

    def market_stats(self):
        stats = {}
        for code, orderbook in self.orderbooks.items():
            volume = orderbook.steps_record['volume'][-1]
            amount = orderbook.steps_record['amount'][-1]
            price = orderbook.steps_record['close'][-1]
            bid = orderbook.bids_sum
            ask = orderbook.asks_sum
            stats[code] = {'price': price, 'amount': amount, 'volume': volume, 'bid': bid, 'ask': ask}
        return stats

    def determine_tick_size(self, base_price):
        if base_price < 10:
            return 0.01
        elif base_price < 50:
            return 0.05
        elif base_price < 100:
            return 0.1
        elif base_price < 500:
            return 0.1
        elif base_price < 1000:
            return 1
        else:
            return 5

class CallMarket(Market):
    def __init__(self, core, interest_rate, interest_period, securities, clear_period, transaction_rate, pre_step):
        self.core = core
        self.is_trading = False
        self.stock_size = 100
        self.interest_rate = interest_rate
        self.interest_period = interest_period
        self.clear_period = clear_period
        self.transaction_rate = transaction_rate
        self.orderbooks = self.set_orderbooks(securities, pre_step)

    def set_orderbooks(self, securities, pre_step):
        orderbooks = {}
        for code, paras in securities.items():
            orderbooks[code] = CallOrderBook(self, code, **paras)
            init_value = orderbooks[code].value

            prices = [init_value]
            values = [init_value]
            for _ in range(pre_step):
                prices.append(round(math.exp(math.log(prices[-1]) + random.gauss(0, 0.005)), 1))
                values.append(round(math.exp(math.log(values[-1]) + random.gauss(0, 0.005)), 1))

            volumes = [int(random.gauss(100, 10)*10) for _ in range(pre_step)]

            orderbooks[code].steps_record["price"] = prices
            orderbooks[code].steps_record["volume"] = volumes
            orderbooks[code].steps_record["amount"] = [price * volume for price, volume in zip(prices, volumes)]
            orderbooks[code].steps_record["value"] = values
        
        return orderbooks

    def step(self):
        # adjust value
        timestep = self.get_time()
        for code, orderbook in self.orderbooks.items():
            if len(orderbook.bids_price) == 0 or len(orderbook.asks_price) == 0 or orderbook.bids_price[0] < orderbook.asks_price[0]:
                updated_info = orderbook.handle_no_match()
                done = True
            else:
                match_price, max_match_volume = orderbook.match_order()
                updated_info = orderbook.fill_orders(match_price, max_match_volume)
                done = False
            
            orderbook.update_record(**updated_info)
            orderbook.clear_orders()
            orderbook.step_summarize()
            self.change_value(code)
            # if timestep > 0 and timestep % 20 == 0:
                # self.random_event(code)
            
        
        if timestep > 0 and timestep % self.interest_period == 0:
            self.issue_interest()
            self.issue_dividends()
        
        return done
    

    def market_stats(self):
        stats = {}
        for code, orderbook in self.orderbooks.items():
            value = orderbook.steps_record['value'][-1]
            volume = orderbook.steps_record['volume'][-1]
            amount = orderbook.steps_record['amount'][-1]
            price = orderbook.steps_record['price'][-1]
            bid = orderbook.steps_record['bid'][-1]
            ask = orderbook.steps_record['ask'][-1]
            stats[code] = {'price': price, 'value': value, 'amount': amount, 'volume': volume, 'bid': bid, 'ask': ask}
        return stats

    def get_current_price(self, code):
        return self.orderbooks[code].current_record['price']

    def get_current_value(self, code):
        return self.orderbooks[code].current_record['value']

    def get_current_value_with_noise(self, code):
        v = random.gauss(0, 0.01)
        real_value = self.orderbooks[code].current_record['value']
        step = random.randint(1, 60)
        values = self.orderbooks[code].steps_record['value'][-1*step::]
        d_v = np.diff(values).sum().item()
        
        # noisy_value = round(math.exp(math.log(real_value) + v), 1)
        noisy_value = real_value + d_v
        return noisy_value

        
        
    def get_base_price(self, code):
        return self.orderbooks[code].steps_record['price'][0]

    def get_base_volume(self, code):
        return self.orderbooks[code].steps_record['volume'][0]