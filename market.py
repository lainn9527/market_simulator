from order_book import OrderBook
from core import Message
from order import LimitOrder, MarketOrder

class Market:
    def __init__(self, securities):
        self.orderbooks = self.build_orderbooks(securities)
        self.raw_orders = list()
        self.core = None
        self.start_time = None
    
    def build_orderbooks(self, securities):
        if len(securities) == 0:
            raise Exception

        return {code: OrderBook(self, code) for code in securities}

    def start(self, core, start_time):
        self.core = core
        self.start_time = start_time
    
    def get_orderbook(self, code):
        if code not in self.orderbooks.keys():
            raise Exception
        return self.orderbooks['code']

    def get_time(self):
        pass

    def best_asks(self, code):
        pass
    
    def best_bids(self, code):
        pass
    
    def inside_asks(self, code, price):
        ob = get_orderbook(code)

    
    def inside_bids(self, code, price):
        pass

    def update_price(self):
        pass

    def step(self):
        pass

    def open_market(self):
        pass

    def close_market(self):
        pass
    
    def send_message(self, message, send_time):
        self.core.send_message(message, send_time)

    def receive_message(self, message):
        if message.receiver != 'market':
            raise Exception
        
        if message.subject == 'LIMIT_ORDER':
            if not isinstance(message.content, LimitOrder):
                raise Exception
            self.get_orderbook(message.content.code).handle_limit_order(message.content, self.get_time())

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

    def generate_order_id(self):
        pass