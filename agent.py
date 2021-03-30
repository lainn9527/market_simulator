from core import Core, Message
from order import LimitOrder, MarketOrder
from datetime import datetime as dt

class Agent:
    def __init__(self, _type, _id, start_cash):
        self.unique_id = _id
        self.type = _type
        self.start_cash = start_cash
        self.signature = f"agent_{self.type}_{self.unique_id}"
        self.core = None
        self.start_time = None
        self.time_scale = None
        self.current_time = None

        # information of markets
        self.orders = None # {'code': [{'order': order, 'placed_time': datetime, 'finished_time': datetime,  'filled_quantity': int, 'filled_amount': float}] }
        self.holdings = {'CASH': start_cash}

        # state flag
        self.is_trading = False
        self.is_open_auction = False
        self.is_close_auction = False
        self.is_continuous_trading = False
    
    def start(self, core, start_time, time_scale):
        self.core = core
        self.start_time = start_time
        self.time_scale = time_scale
        # ready to go

    def step(self):
        self.current_time += 1

    def place_order(self, _type, code, bid_or_ask, volume, price):
        # check valid
        if _type == 'limit':
            order = LimitOrder(self.signature, code, 'LIMIT', bid_or_ask, volume, price)
            msg = Message('MARKET', 'LIMIT_ORDER', self.signature, 'market', order)
        elif _type == 'market':
            order = MarketOrder(self.signature, code, 'MARKET', bid_or_ask, volume)
            msg = Message('MARKET', 'MARKET_ORDER', self.signature, 'market', order)
        elif _type == 'auction':
            order = LimitOrder(self.signature, code, 'LIMIT', bid_or_ask, volume, price)
            msg = Message('MARKET', 'AUCTION_ORDER', self.signature, 'market', order)
        else:
            raise Exception
        
        self.orders['code'].append({'order': order, 'placed_time': None, 'finished_time': None, 'filled_quantity': 0, 'filled_amount': 0.0})
        self.core.send_message(msg, self.current_time())

    def receive_message(self, message):
        if message.receiver != self.signature:
            raise Exception('Wrong receiver!')

        message_subject = ['OPEN_SESSION', 'CLOSE_SESSION', 'OPEN_AUCTION', 'CLOSE_AUCTION', 'OPEN_CONTINUOUS_TRADING', 'STOP_CONTINUOUS_TRADING', 'ORDER_PLACED', 'ORDER_CANCELLED', 'ORDER_INVALIDED', 'ORDER_FILLED', 'ORDER_FINISHED']

        if message.subject == 'OPEN_SESSION':
            # receive time and ready to palce order
            self.current_time = message.content

        elif message.subject == 'CLOSE_SESSION':
            # receive the daily info of market: daily_info[code] = {'date': date in isoformat, 'order': self.orders, 'bid': self.bids, 'ask': self.asks, 'stats': self.stats }
            pass

        elif message.subject == 'OPEN_AUCTION':
            # receive base prices when open, nothing when close
            self.is_trading = True
            if message.content == None:
                self.is_close_auction = True
            else:
                self.is_open_auction = True
            

        elif message.subject == 'CLOSE_AUCTION':
            # receive open price when open, close price when close
            # timestep++
            if 'open' in message.content.keys():
                self.is_open_auction = False
            elif 'close' in message.content.keys():
                self.is_trading = False
                self.is_close_auction = False

            # stop placing order in 08:59:59.999 or 13:29:29.999
            # steps by hand
            self.current_time += 1

        elif message.subject == 'OPEN_CONTINUOUS_TRADING':
            self.is_continuous_trading = True
            # ready to place order in continuous trading (09:00:00)

        elif message.subject == 'STOP_CONTINUOUS_TRADING':
            self.is_continuous_trading = False

            # stop placing order in continuous trading (13:24:59.999)
            # steps to 13:25:00 by hand
            self.current_time += 1

        elif message.subject == 'ORDER_PLACED':
            # receive order info = {'order_id': order_id, 'code': order.code, 'price': order.price, 'quantity': order.quantity}
            # check if the order is transformed from market order
            if isinstance(message.content, MarketOrder):
                self.orders['code'][-1]['order'] = LimitOrder.from_market_order(self.last_orders['code'], message.content['price'])
            self.orders['code']['placed_time'] = message.content['time']

            self.log_event('ORDER_PLACED', self.orders['code'][-1])


        elif message.subject == 'ORDER_FILLED':
            # receive the transactions (partial filled) of the order = {'code': code, 'price': price, 'quantity': quantity}
            self.orders[message.content['code']][-1]['filled_quantity'] += message.content['quantity']
            self.orders[message.content['code']][-1]['filled_amount'] += message.content['price'] * message.content['quantity']
            
            self.log_event('ORDER_FILLED', {'price': message.content['price'], 'quantity': message.content['quantity']})


        elif message.subject == 'ORDER_FINISHED':
            # receive the finisged order info = {'price': round(total_amount/total_quantity, 2),'quantity': total_quantity}
            if self.orders[message.content['code']][-1]['order'].quantity != self.orders[message.content['code']][-1]['filled_quantity']:
                raise Exception("The quantity hasn't fulfilled")
            self.orders[message.content['code']][-1]['finished_time'] = message.content['time']

            self.log_event('ORDER_FINISHED', self.orders[message.content['code']][-1])
            
        else:
            raise Exception(f"Invalid subject for agent {self.signature}")

    def current_price(self, bid_or_ask, code, number):
        return self.core.best_bids(code, number) if bid_or_ask == 'BID' else self.core.best_asks(code, number)

    def log_event(self, event_type, event):
        print("Agent: {self.unique_id}, Event type: {event_type}, Event: {event}")