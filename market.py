from order_book import OrderBook
from core import Message
from order import LimitOrder, MarketOrder

class Market:
    def __init__(self, security_values):
        self.orderbooks = self.build_orderbooks(security_values)
        self.raw_orders = list()
        self.core = None
        self.start_time = None
        self.current_time = None
    
    def build_orderbooks(self, security_values):
        if len(security_values) == 0:
            raise Exception
        
        # initiate prices of securities

        return {code: OrderBook(self, code, value) for code, value in security_values}

    def start(self, core, start_time):
        self.core = core
        self.start_time = start_time
        
    
    def get_orderbook(self, code):
        if code not in self.orderbooks.keys():
            raise Exception
        return self.orderbooks['code']

    def get_time(self):
        return self.current_time

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

    def close_auction(self):
        # match price of the bid and ask with max volume
        for orderbook in self.orderbooks:
            if len(orderbook.bids) == 0 or len(orderbook.asks) == 0:
                print('No quote!')
                pass
            if orderbook.bids[0][0] < orderbook.asks[0][0]:
                print('No match')
                pass

            best_bid = orderbook.bids[0][0]
            match_index = -1
            for i, pvo in enumerate(orderbook.asks):
                if best_bid == pvo[0]:
                    match_index = i
                    break
            if match_index == -1:
                raise Exception('Liquidity Error: No successive quotes')

            accum_bid = [pvo[1] for pvo in orderbook.bids]
            accum_ask = [pvo[1] for pvo in orderbook.asks]
            for i in range(1, accum_bid):
                accum_bid[i] = accum_bid[i] + accum_bid[i-1]
            for i in range(1, accum_ask):
                accum_ask[i] = accum_ask[i] + accum_ask[i-1]
            
            i = 0
            j = match_index
            max_match_volume = 0
            match_bid_index = -1
            match_ask_index = -1
            while i < len(orderbook.bids) and j > 0:
                match = min(accum_bid[i], accum_ask[j])
                if match > max_match_volume:
                    max_match_volume = match
                    match_bid_index = i
                    match_ask_index = j
                i += 1
                j -= 1
            
            match_price = orderbook.bids[match_bid_index][0]
            # fill the orders
            remain_quantity = max_match_volume
            for order_id, quantity in orderbook.bids[match_bid_index][2]:
                orderbook.fill_order(order_id, self.get_time(), match_price, min(remain_quantity, quantity))
                remain_quantity -= quantity
            

        


    def open_session(self, open_time, timestep):
        '''
            1. set the base price for auction and notify all agents
            2. simulate the auction from 0830 to 0900 to decide open price and volume
            3. record the open price and notify all agents
            4. ready for the continuous trading
            TODO: auction
        '''
        self.current_time = open_time

        # set the base price of securities
        base_prices = {}
        for orderbook in self.orderbooks:
            base_prices[orderbook.code] = orderbook.set_base_price()
        
        # send open time and base price to all agents and start the auction
        msg = Message('ALL_AGENTS', 'OPEN_AUCTION', 'market', 'agents', [self.current_time, base_prices])

        # little trick to force the core notifying agents
        # TODO: add function 'announce' in the core to represent the market announcement (like open market)
        self.core.handle_message(msg)

        
        

    def close_session(self):
        '''
            Works to be done at the end of a market day:
            1. TODO: call auction in the close session
            2. clear the order and notify the agents
            3. summarize trading information today
            4. return the information to the core to notify the agents
        '''
        # auction to get the close price
        daily_info = {}
        close_price = {}
        for orderbook in self.orderbooks:
            # get the daily info
            daily_info[orderbook.code] = 0


            
    def send_message(self, message, send_time):
        self.core.send_message(message, send_time)

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