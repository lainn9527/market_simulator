from order_book import OrderBook
from core import Message
from order import LimitOrder, MarketOrder
from datetime import time
class Market:
    def __init__(self, security_values):
        self.orderbooks = self.build_orderbooks(security_values)
        self.core = None
        self.start_time = None
        self.time_scale = None
        self.current_time = None
    
    def build_orderbooks(self, security_values):
        if len(security_values) == 0:
            raise Exception
        
        # initiate prices of securities

        return {code: OrderBook(self, code, value) for code, value in security_values}

    def start(self, core, start_time, time_scale):
        self.core = core
        self.start_time = start_time
        self.time_scale = time_scale
        
    
    def get_orderbook(self, code):
        if code not in self.orderbooks.keys():
            raise Exception
        return self.orderbooks['code']

    def get_time(self):
        return self.current_time

    def best_asks(self, code, number):
        if len(self.get_orderbook(code).asks) < number:
            number = len(self.get_orderbook(code).asks)
        return [self.get_time(), [ [item[0], item[1]] for item in self.get_orderbook(code).asks[:number] ]]
    
    def best_bids(self, code, number):
        if len(self.get_orderbook(code).bids) < number:
            number = len(self.get_orderbook(code).bids)
        return [self.get_time(), [ [item[0], item[1]] for item in self.get_orderbook(code).bids[:number] ]]
    
    def inside_asks(self, code, price):
        ob = get_orderbook(code)

    
    def inside_bids(self, code, price):
        pass

    def update_price(self):
        pass

    def step(self):
        self.current_time += self.time_scale

    def open_session(self, open_time):
        self.current_time = open_time
        msg = Message('ALL_AGENTS', 'OPEN_SESSION', 'market', 'agents', self.get_time())
        self.announce(msg, self.get_time())
        
        

    def close_session(self):
        '''
            Works to be done at the end of a market day:
            1. clear the order and notify the agents
            2. summarize trading information today
            3. send the daily stats to agents
        '''


        daily_info = {}
        close_price = {}
        for orderbook in self.orderbooks:
            # get the daily info
            daily_info[orderbook.code] = orderbook.daily_summarize()

        self.announce(Message('ALL_AGENTS', 'CLOSE_SESSION', 'market', 'agents', daily_info))

    def start_auction(self):
        # if open, set the base price for auction and notify all agents
        if self.get_time().time() == time(hour = 8, minute = 30, second = 0):
            base_prices = {}
            for orderbook in self.orderbooks:
                base_prices[orderbook.code] = orderbook.set_base_price()
            msg = Message('ALL_AGENTS', 'OPEN_AUCTION', 'market', 'agents', base_prices)
        elif self.get_time().time() == time(hour = 13, minute = 25, second = 0):
            msg = Message('ALL_AGENTS', 'OPEN_AUCTION', 'market', 'agents', None)
        # send open time and base price to all agents and start the auction
        self.announce(msg, self.get_time())

    def close_auction(self):
        self.step()
        price = {}
        for orderbook in self.orderbooks:
            price[orderbook.code] = orderbook.handle_auction()
        msg = Message('ALL_AGENTS', 'CLOSE_AUCTION', 'market', 'agents', price)
        self.announce(msg, self.get_time())

    def start_continuous_trading(self):
        msg = Message('ALL_AGENTS', 'OPEN_CONTINUOUS_TRADING', 'market', 'agents', None)
        self.announce(msg, self.get_time())

    def close_continuous_trading(self):
        self.step()
        msg = Message('ALL_AGENTS', 'STOP_CONTINUOUS_TRADING', 'market', 'agents', None)
        self.announce(msg, self.get_time())
            
    def send_message(self, message, send_time):
        self.core.send_message(message, send_time)

    def announce(self, message, send_time):
        self.core.announce(message, send_time)

    def receive_message(self, message):
        if message.receiver != 'market':
            raise Exception
        
        if message.subject == 'LIMIT_ORDER':
            if not isinstance(message.content, LimitOrder):
                raise Exception
            self.get_orderbook(message.content.code).handle_limit_order(message.content, self.get_time(), True)

        elif message.subject == 'AUCTION_ORDER':
            if not isinstance(message.content, LimitOrder):
                raise Exception
            self.get_orderbook(message.content.code).handle_limit_order(message.content, self.get_time(), False)

        elif message.subject == 'MARKET_ORDER':
            if not isinstance(message.content, MarketOrder):
                raise Exception
            self.get_orderbook(message.content.code).handle_market_order(message.content, self.get_time())

        
        elif message.subject == 'CANCEL_ORDER':
            self.handle_market_order(message.content)
        
        elif message.subject == 'MODIFY_ORDER':
            self.modify_order(message.content)

        else:
            raise Exception
    

    def cancel_order(self, order):
        pass

    def modify_order(self, order):
        pass