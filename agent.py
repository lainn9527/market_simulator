from message import Message
from order import LimitOrder, MarketOrder
from datetime import datetime, timedelta, time

class Agent:
    num_of_agent = 0

    def __init__(self, _type, start_cash = 10000000, security_unit = 1000):
        Agent.add_counter()
        self.type = _type
        self.start_cash = start_cash
        self.security_unit = security_unit
        self.unique_id = f"agent_{self.type}_{self.get_counter()}"
        self.core = None
        
        
        '''
        Information of markets
        orders: {security code: [order_id] }
        holdings: {security code: volume}
        '''
        self.holdings = {'CASH': start_cash}
        self.orders = {}
        self.orders_history = {}

        # state flag
        self.is_trading = False
        self.total_timestep = None
    
    def start(self, core, securities):
        self.core = core
        self.holdings.update({code: 0 for code in securities})
        self.orders.update({code: [] for code in securities})
        self.orders_history.update({code: [] for code in securities})
        # ready to go
        self.log_event('AGENT_PREPARED', 'Ready to go!')

    def step(self):
        pass

    def receive_message(self, message):
        if message.receiver != self.unique_id and message.receiver != 'agents':
            raise Exception('Wrong receiver!')

        message_subject = ['OPEN_SESSION', 'CLOSE_SESSION', 'OPEN_AUCTION', 'CLOSE_AUCTION', 'OPEN_CONTINUOUS_TRADING', 'STOP_CONTINUOUS_TRADING', 'ORDER_PLACED', 'ORDER_CANCELLED', 'ORDER_INVALIDED', 'ORDER_FILLED', 'ORDER_FINISHED']

        if message.subject == 'OPEN_SESSION':
            # receive time and ready to palce order
            self.handle_open()

        elif message.subject == 'CLOSE_SESSION':
            # receive the daily info of market: daily_info[code] = {'date': date in isoformat, 'order': self.orders, 'bid': self.bids, 'ask': self.asks, 'stats': self.stats }
            self.handle_close()

        elif message.subject == 'ORDER_PLACED':
            # check if the order is transformed from market order
            self.orders[message.content['code']].append(message.content['order_id'])            
            # self.log_event('ORDER_PLACED', self.orders[order.code][order.order_id])


        elif message.subject == 'ORDER_FILLED':
            order_record = self.core.get_order_record(message.content['code'], message.content['order_id'])
            if message.content['bid_or_ask'] == 'BID':
                self.holdings[message.content['code']] += message.content['quantity']
                self.holdings['CASH'] += (order_record.order.price - message.content['price']) * message.content['quantity']
            
            elif message.content['bid_or_ask'] == 'ASK':
                self.holdings['CASH'] += message.content['price'] * message.content['quantity']
            
            
            self.log_event('ORDER_FILLED', {'price': message.content['price'], 'quantity': message.content['quantity']})


        elif message.subject == 'ORDER_FINISHED':
            for i, order_id in enumerate(self.orders[message.content['code']]):
                if order_id == message.content['order_id']:
                    finished_order_id = self.orders[message.content['code']].pop(i)
            self.orders_history[message.content['code']].append(finished_order_id)
            
            self.log_event('ORDER_FINISHED', self.orders[message.content['code']][message.content['order_id']])
        
        elif message.subject == 'ORDER_INVALIDED':
            pass
        else:
            raise Exception(f"Invalid subject for agent {self.unique_id}")

    def place_limit_bid_order(self, code, volume, price):
        cost = volume * price
        if cost > self.holdings['CASH']:
            raise Exception("Not enough money")
        self.holdings['CASH'] -= cost
        order = LimitOrder(self.unique_id, code, 'LIMIT', 'BID', volume, price)
        msg = Message('MARKET', 'LIMIT_ORDER', self.unique_id, 'market', {'order': order})
        
        self.core.send_message(msg)

    def place_limit_ask_order(self, code, volume, price):
        if volume > self.holdings[code]:
            raise Exception(f"Not enough {code} shares")
        self.holdings[code] -= volume
        order = LimitOrder(self.unique_id, code, 'LIMIT', 'ASK', volume, price)
        msg = Message('MARKET', 'LIMIT_ORDER', self.unique_id, 'market', {'order': order})
        
        self.core.send_message(msg)



    def place_market_bid_order(self, code, volume):
        pass

    def place_market_ask_order(self, code, volume):
        pass
    
    def modify_order(self):
        pass

    def cancel_order(self):
        pass

    def current_price(self, bid_or_ask, code, number):
        return self.core.best_bids(code, number) if bid_or_ask == 'BID' else self.core.best_asks(code, number)

    def log_event(self, event_type, event):
        # print(f"Agent: {self.unique_id}, Event type: {event_type}, Event: {event}")
        return
    
    @classmethod
    def add_counter(cls):
        cls.num_of_agent += 1

    @classmethod
    def get_counter(cls):
        return cls.num_of_agent
    
    def generate_id(self):
        pass
        
    def handle_open(self):
        pass

    def schedule_auction(self):
        pass
    
    def schedule_continuous_trading(self):
        pass
    
    def handle_close(self):
        pass

    def get_order(self):
        pass
    