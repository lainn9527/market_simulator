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
        for orderbook in self.orderbooks.values():
            orderbook.step_summarize()
        
        
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
        
        elif message.subject == 'CANCEL_ORDER':
            pass
        
        elif message.subject == 'MODIFY_ORDER':
            pass

        else:
            raise Exception

    def get_order_record(self, code, order_id):
        return self.orderbooks[code].orders[order_id]

    def get_current_price(self, code):
        return self.orderbooks[code].current_record['price']

    def get_records(self, step = 1):
        return self.orderbooks[code].steps_record[:-1*(step+1):-1]

    def get_best_bids(self, code, number = 1):
        return [{'price': price, 'volume': self.orderbooks[code].bids_volume[price]} for price in self.orderbooks[code].bids_price[:number]]

    def get_best_asks(self, code, number = 1):
        return [{'price': price, 'volume': self.orderbooks[code].asks_volume[price]} for price in self.orderbooks[code].bids_price[:number]]
    
    def get_tick_size(self, code):
        return self.orderbooks[code].tick_size

    def get_securities(self):
        return list(self.orderbooks.keys())

    def get_time(self):
        return self.core.timestep

    def market_stats(self):
        stats = {}
        for code, orderbook in self.orderbooks.items():
            amount = orderbook.steps_record[-1]['volume']
            volume = orderbook.steps_record[-1]['amount']
            average_price = orderbook.steps_record[-1]['average']
            stats[code] = {'average price': average_price, 'amount': amount, 'volume': volume}
        return stats

    def determine_tick_size(self, base_price):
        if base_price < 10:
            return 0.01
        elif base_price < 50:
            return 0.05
        elif base_price < 100:
            return 0.1
        elif base_price < 500:
            return 0.5
        elif base_price < 1000:
            return 1
        else:
            return 5