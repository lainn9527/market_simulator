from order_book import OrderBook
from message import Message
from order import LimitOrder, MarketOrder
from datetime import time, timedelta
class Market:
    def __init__(self, security_values):
        self.orderbooks = {code: OrderBook(self, code, value) for code, value in security_values.items()}
        self.core = None
        self.start_time = None
        self.time_scale = None
        self.current_time = None


    def build_orderbooks(self, security_values):
        if len(security_values) == 0:
            raise Exception
        
        # initiate prices of securities

        return {code: OrderBook(self, code, value) for code, value in security_values.items()}


    def start(self, core, start_time, time_scale):
        self.core = core
        self.start_time = start_time
        self.time_scale = time_scale
        
    
    def get_orderbook(self, code):
        if code not in self.orderbooks.keys():
            raise Exception
        return self.orderbooks[code]

    def get_time(self):
        return self.current_time

    def best_asks(self, code, number = 1):
        return [{'price': item[0], 'volume': item[1]} for item in self.get_orderbook(code).asks[self.get_orderbook(code).best_ask_index:self.get_orderbook(code).best_ask_index + number]]
    
    def best_bids(self, code, number = 1):
        return [{'price': item[0], 'volume': item[1]} for item in self.get_orderbook(code).bids[max(self.get_orderbook(code).best_bid_index + 1 - number, 0):self.get_orderbook(code).best_bid_index + 1]]
    

    def step(self):
        self.current_time += timedelta(seconds = self.time_scale)

    def open_session(self, open_time, total_timestep):
        self.current_time = open_time
        # determine the price list of orderbook
        for _, orderbook in self.orderbooks.items():
            orderbook.set_price_list()
            # orderbook

        # notify agents
        msg = Message('ALL_AGENTS', 'OPEN_SESSION', 'market', 'agents', {'time': self.get_time(), 'total_timestep': total_timestep})
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
        for code, orderbook in self.orderbooks.items():
            # get the daily info
            daily_info[code] = orderbook.daily_summarize()

        self.announce(Message('ALL_AGENTS', 'CLOSE_SESSION', 'market', 'agents', daily_info), self.get_time())

    def start_auction(self, timestep):
        # send open time to all agents and start the auction
        msg = Message('ALL_AGENTS', 'OPEN_AUCTION', 'market', 'agents', {'timestep': timestep})
        self.announce(msg, self.get_time())

    def close_auction(self):
        self.step()
        for code, orderbook in self.orderbooks.items():
            orderbook.handle_auction()

        msg = Message('ALL_AGENTS', 'CLOSE_AUCTION', 'market', 'agents', None)
        self.announce(msg, self.get_time())

    def start_continuous_trading(self, timestep):
        msg = Message('ALL_AGENTS', 'OPEN_CONTINUOUS_TRADING', 'market', 'agents', {'timestep': timestep})
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
            self.get_orderbook(message.content.code).handle_limit_order(message.content, False)

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

    def get_securities(self):
        return list(self.orderbooks.keys())

    def get_price_info(self, code):
        return self.get_orderbook(code).price_info

    def market_stats(self):
        amount = 0
        volume = 0
        for code, orderbook in self.orderbooks.items():
            amount = orderbook.stats['amount']
            volume = orderbook.stats['volume']
        return {'amount': amount, 'volume': volume}

    def determine_tick_size(self):
        if self.value < 10:
            return 0.01
        elif self.value < 50:
            return 0.05
        elif self.value < 100:
            return 0.1
        elif self.value < 500:
            return 0.5
        elif self.value < 1000:
            return 1
        else:
            return 5