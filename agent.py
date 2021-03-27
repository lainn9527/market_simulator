from core import Core, Message
from order import LimitOrder, MarketOrder
from datetime import datetime as dt

class Agent:
    def __init__(self, _type, _id):
        self.unique_id = _id
        self.type = _type
        self.signature = f"agent_{self.type}_{self.unique_id}"
        self.core = None
        self.start_time = None
        self.time_scale = None
        self.current_time = None

    def start(self, core, start_time, time_scale):
        self.core = core
        self.start_time = start_time
        self.time_scale = time_scale
        # ready to go

    def step(self):
        pass

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
        self.core.send_message(msg, self.current_time())

    def receive_message(self, message):
        if message.receiver != self.signature:
            raise Exception('Wrong receiver!')

        message_subject = ['OPEN_SESSION', 'CLOSE_SESSION', 'OPEN_AUCTION', 'CLOSE_AUCTION', 'OPEN_CONTINUOUS_TRADING', 'STOP_CONTINUOUS_TRADING', 'ORDER_PLACED', 'ORDER_CANCELLED', 'ORDER_INVALIDED', 'ORDER_FILLED', 'ORDER_FINISHED']
        if message.subject == 'OPEN_SESSION':
            pass
        elif message.subject == 'CLOSE_SESSION':
            pass
        elif message.subject == 'OPEN_AUCTION':
            pass
        elif message.subject == 'CLOSE_AUCTION':
            pass
        elif message.subject == 'OPEN_CONTINUOUS_TRADING':
            pass
        elif message.subject == 'STOP_CONTINUOUS_TRADING':
            pass
        elif message.subject == 'ORDER_PLACED':
            pass
        elif message.subject == 'ORDER_FILLED':
            pass
        elif message.subject == 'ORDER_FINISHED':
            pass
        else:
            raise Exception(f"Invalid subject for agent {self.signature}")


