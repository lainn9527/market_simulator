from order_book import OrderBook
from message import Message
from order import LimitOrder, MarketOrder
from datetime import time, timedelta
class Market:
    def __init__(self, security_values):
        self.orderbooks = {code: OrderBook(self, code, value) for code, value in security_values.items()}
        self.core = None
        self.is_trading = False

    def start(self, core):
        self.core = core
    
    def step(self):
        pass
        
    def open_session(self):
        # determine the price list of orderbook
        for orderbook in self.orderbooks.values():
            orderbook.set_price()

        # notify agents
        msg = Message('ALL_AGENTS', 'OPEN_SESSION', 'market', 'agents')
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
        
        elif message.subject == 'CANCEL_ORDER':
            pass
        
        elif message.subject == 'MODIFY_ORDER':
            pass

        else:
            raise Exception

    def get_order_record(self, code, order_id):
        return self.orderbooks[code].orders[order_id]

    def get_orderbook(self, code):
        if code not in self.orderbooks.keys():
            raise Exception
        return self.orderbooks[code]

    def get_best_asks(self, code, number = 1):
        return [{'price': price, 'volume': self.orderbooks[code].asks_volume[price]} for price in self.orderbooks[code].bids_price[:number]]
    
    def get_best_bids(self, code, number = 1):
        return [{'price': price, 'volume': self.orderbooks[code].bids_volume[price]} for price in self.orderbooks[code].bids_price[:number]]
    
    def get_securities(self):
        return list(self.orderbooks.keys())

    def get_price_info(self, code):
        return self.orderbooks[code].price_info

    def get_time(self):
        return self.core.timestep

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