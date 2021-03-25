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

        return {code: OrderBook(code) for code in securities}

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

    def receive_message(self, message):
        if message.receiver != 'market':
            raise Exception
        
        if message.subject == 'LIMIT_ORDER':
            self.handle_limit_order(message.content)

        elif message.subject == 'MARKET_ORDER':
            self.handle_market_order(message.content)
        
        elif message.subject == 'CANCEL_ORDER':
            self.handle_market_order(message.content)
        
        elif message.subject == 'MODIFY_ORDER':
            self.modify_order(message.content)

        else:
            raise Exception
    
    
    def handle_limit_order(self, order):
        if not isinstance(order, LimitOrder):
            raise Exception
        
        messages = []
        # add the limit order to the orderbook
        order_id = self.get_orderbook(order.code).add_order(order, self.get_time())
        # send msg to agent

        # match the order
        finished, matched_orders = self.get_orderbook(order.code).match_order(order_id)
        

        remain_quantity = order.quantity

        self._handle_limit_order(order)
        finished, matched_orders = self.get_orderbook(order.code).match_order(order.bid_or_ask, order.price, order.quantity)
        if finished:
            pass

        # check matching order

        while remain_quantity > 0:
            # check if price is inside
            if order.bid_or_ask == 'BID' and order.price >= self.best_asks(order.code):
                best_pv = self.best_asks(order.code)
            elif order.bid_or_ask == 'ASK' and order.price <= self.best_bids(order.code):
                best_pv = self.best_bids(order.code)
            else:
                break
            # if remained quantity is smaller, match the remained quantity
            # if larger, match the quantity on the orderbook
            if remain_quantity >= best_pv[1]:
                remain_quantity -= best_pv[1]
                pv = best_pv
            elif remain_quantity < best_pv[1]:
                pv = (best_pv[0], remain_quantity)
            
            # match by price and volumn on orderbook
            matched_orders += self.get_orderbook(order.code).match_order(order.bid_or_ask, pv[0], pv[1])
        
        # place remained shares of order
        if remain_quantity > 0:
            order_info = self.get_orderbook(order.code).add_order(order.bid_or_ask, order.price, remain_quantity)
        
        # TODO
        # send message back to the agents
             

    def handle_market_order(self, order):
        pass

    def cancel_order(self, order):
        pass

    def modify_order(self, order):
        pass

    def generate_order_id(self):
        pass