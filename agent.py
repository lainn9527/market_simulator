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

    def start(self, core, start_time):
        self.core = core
        self.start_time = start_time
        # ready to go

    def step(self):
        pass

    def place_order(self, _type, code, bid_or_ask, volumn, price):
        # check valid
        if _type == 'limit':
            order = LimitOrder(self.signature, code, bid_or_ask, volumn, price)
            msg = Message('MARKET', self.signature, 'limit', order)
        elif _type == 'market':
            order = MarketOrder(self.signature, code, bid_or_ask, volumn)
            msg = Message('MARKET', self.signature, 'market', order)
        else:
            raise Exception
        
        self.core.send_message(msg, dt.now().timestamp())

    def receive_message(self, message):
        pass
