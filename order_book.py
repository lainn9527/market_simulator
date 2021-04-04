import pandas as pd
import datetime
import math
from order import LimitOrder, MarketOrder
from dataclasses import dataclass
from typing import Any, List, Tuple
from message import Message

class OrderBook:
    def __init__(self, market, code, value):
        '''
            history: [{
                'date': in isoformat (e.g. 2021-03-08),
                'order': orders,
                'bid': bids,
                'ask': asks,
                'stats': stats
            }]
            orders: {
                'Order ID': {
                    'order': Order,
                    'time': Time,
                    'state': bool,
                    'transactions': ['time', 'price', 'quantity'],
                    'modifications': [],
                    'cancellation': bool
                }
            }
            code: code of security
            bids: [[price, volume, [[order_id, quantity]]]]
            asks: [[price, volume, [[order_id, quantity]]]
            stats: {
                'base': float,
                'value: float
                'amount': float,
                'volume': int,
                'bid': int,
                'ask': int,
                'average': float,
                'open': float,
                'high': float,
                'low': float,
                'close': float
            }
        '''
        self.market = market
        self.code = code
        self.value = value
        self.history = []

        self.orders = None
        self.bids = None
        self.asks = None
        self.best_bid_index = None
        self.best_ask_index = None
        self.num_of_order = None
        self.stats = None
        self.price_info = None
        self.reset()


    def reset(self):
        self.orders = dict()
        self.num_of_order = 0
        self.price_info = {
            'up_limit': 0.0,
            'low_limit': 0.0,
            'tick_size': 0.0,
            'base': 0.0,
        }
        self.stats = {
            'up_limit': 0.0,
            'low_limit': 0.0,
            'tick_size': 0.0,
            'base': 0.0,
            'value': 0.0,
            'amount': 0.0,
            'volume': 0,
            'bid': 0,
            'ask': 0,
            'average': 0.0,
            'open': 0.0,
            'high': 0.0,
            'low': 0.0,
            'close': 0.0
        }
    

    def set_price_list(self):
        # base price for call auction in the open session
        # use the close price of previous day as the base price and if it's the first day, use the fundamental value instead

        if len(self.history) == 0:
            base_price = self.value
        else:
            base_price = self.history[-1]['stats']['close']

        # initalize the valid price list
        tick_size = self.determine_tick_size(base_price)
        valid_ticks = math.floor((base_price * 0.1) / tick_size)
        valid_price = [round(base_price + tick_size * i, 2) for i in range(-1 * valid_ticks, valid_ticks + 1, 1)]
        low_limit = valid_price[0]
        up_limit = valid_price[-1]
        self.bids = [valid_price.copy(), [0]*len(valid_price), []]
        self.asks = [valid_price.copy(), [0]*len(valid_price), []]

        # point to the base price
        self.best_bid_index = self.best_ask_index = valid_ticks        


        self.stats['base'] = base_price
        self.stats['up_limit'] = up_limit
        self.stats['low_limit'] = low_limit
        self.stats['tick_size'] = tick_size

    
    def handle_auction(self):
        time = self.market.get_time()

        # locate the best bid at asks
        best_bid = self.bids[self.best_bid_index][0]
        if self.best_bid_index < self.best_ask_index:
            raise Exception('Liquidity Error: No matched auction order.')

        # construct accumulated volume of bids and asks
        accum_bid_ask = [[pvo[1] for pvo in self.bids], [pvo[1] for pvo in self.asks]]

        for i in range(len(accum_bid_ask[0]) - 2, -1, -1):
            accum_bid_ask[0][i] += accum_bid_ask[0][i+1]
        for i in range(1, len(accum_bid_ask[1])):
            accum_bid_ask[1][i] += accum_bid_ask[1][i-1]
        
        max_match_volume = 0
        max_match_index = -1
        for i in range(len(accum_bid_ask[0])):
            if max_match_volume < min(accum_bid_ask[0][i], accum_bid_ask[1][i]):
                max_match_volume = min(accum_bid_ask[0][i], accum_bid_ask[1][i])
                max_match_index = i
        
        match_price = self.bids[max_match_index][0]

        # fill the orders
        # bids: [[price, volume, [[order_id, quantity]]]]
        for pvos in [self.bids[-1:max_match_index-1:-1], self.asks[:max_match_index+1]]:
            remain_quantity = max_match_volume
            for pvo in pvos:
                if match_price != pvo[0] or remain_quantity == pvo[1]:
                    while len(pvo[2]) != 0:
                        self.fill_order(pvo[2][0][0], pvo[0], pvo[2][0][1])
                        pvo[2].pop(0)
                    remain_quantity -= pvo[1]
                    pvo[1] = 0
                else:
                    pvo[1] -= remain_quantity
                    while remain_quantity > pvo[2][0][1]:
                        self.fill_order(pvo[2][0][0], pvo[0], pvo[2][0][1])
                        remain_quantity -= pvo[2][0][1]
                        pvo[2].pop(0)
                    self.fill_order(pvo[2][0][0], pvo[0], pvo[2][0][1])
                    pvo[2][0][1] -= remain_quantity

        # update best bid/ask
        self.best_bid_index = max_match_index if self.bids[max_match_index][1] != 0 or max_match_index == 0 else max_match_index - 1
        self.best_ask_index = max_match_index if self.asks[max_match_index][1] != 0 or max_match_index == len(self.asks) - 1 else max_match_index + 1

        # update the stats
        updated_info = {
            'amount': match_price * max_match_volume,
            'volume': max_match_volume
        }
        if time == datetime.time(hour = 9, minute = 0, second = 0):
            # auction in open
            updated_info['open'] = match_price
        elif time == datetime.time(hour = 13, minute = 30, second = 0):
            # auction in close
            updated_info['close'] = match_price        
        else:
            raise Exception

        self.update_stats(**updated_info)


    def handle_limit_order(self, order, matching) -> str:
        '''
            Match the limit order and fill the matched order.
            Step:
                1. add the limit order to the orderbook
                2. notify the agent of order placement
                3. match the limit order with best price and fill the matched order
                4. after there is no match pair, fill the limit order
                5. update the bid/ask price
                6. update the statics of the orderbook
            For order in auction, matching in step 3 and 4 is needless. The matching flag will control it.
        '''

        order_id = self._generate_order_id()
        time = self.market.get_time()
        # add the order to the orderbook
        self.orders[order_id] = {
            'order': order,
            'time': time,
            'state': False,
            'transactions': [],
            'modifications': [],
            'cancellation': False
        }

        # send message to the orderer
        self.market.send_message(
            Message('AGENT', 'ORDER_PLACED', 'market', order.orderer, {'order_id': order_id, 'time': time, 'code': order.code, 'price': order.price, 'quantity': order.quantity}),
            time
        )

        remain_quantity = order.quantity
        transaction_amount = 0
        transaction_quantity = 0

        if not matching:
            self.add_order(order_id, order.bid_or_ask, order.price, order.quantity)
        # insert the order if order price is within best price or outside the price
        # else start matching
        elif order.bid_or_ask == 'BID' and (order.price <= self.bids[self.best_bid_index] or order.price < self.asks[self.best_ask_index]):
            self.add_order(order_id, order.bid_or_ask, order.price, order.quantity)
        elif order.bid_or_ask == 'ASK' and (order.price >= self.asks[self.best_ask_index] or order.price > self.bids[self.best_bid_index]):
            self.add_order(order_id, order.bid_or_ask, order.price, order.quantity)
        else:
            # start matching
            transaction_quantity, transaction_amount = self.match_order(order.bid_or_ask, order.price, order.quantity)
            
            # if there is any match, fill the limit order
            if transaction_quantity > 0:
                self.fill_order(order_id, round(transaction_amount/transaction_quantity, 2), transaction_quantity)

            # update bid/ask if the limit order is partially filled and placed order
            if transaction_quantity != order.quantity:
                self.add_order(order_id, order.bid_or_ask, order.price, order.quantity - transaction_quantity)
        
        # update the stats
        updated_info = {
            'amount': transaction_amount,
            'volume': transaction_quantity,
            'average': round(self.stats['amount'] / (self.stats['volume'] + 1e-6), 2),
            'high': order.price if order.price > self.stats['high'] else self.stats['high'],
            'low': order.price if order.price < self.stats['low'] else self.stats['low'],
            'bid': order.quantity - transaction_quantity if order.bid_or_ask == 'BID' else (-1*transaction_quantity),
            'ask': order.quantity - transaction_quantity if order.bid_or_ask == 'ASK' else (-1*transaction_quantity)
        }

        self.update_stats(**updated_info)
        
        return order_id            

    
        
    def handle_market_order(self, order, time):
        time = self.market.get_time()
        
        # assume there is no liquidity issue
        # add the limit order in the best price in the market
        
        price = self.stats['up_limit'] if order.bid_or_ask == 'BID' else self.stats['low_limit']
        transaction_quantity, transaction_amount = self.match_order(order.bid_or_ask, price, order.quantity)
        transaction_price = round(transaction_amount / transaction_quantity, 2)
        if transaction_quantity != order.quantity:
            raise Exception("Liquidity error: The market order isn't finished.")

        order_id = self._generate_order_id()
        limit_order = LimitOrder.from_market_order(order, transaction_price)

        self.orders[order_id] = {
            'order': limit_order,
            'time': time,
            'state': True,
            'transactions': [[time, transaction_price, transaction_quantity]],
            'modifications': [],
            'cancellation': False
        }

        self.market.send_message(
            Message('AGENT', 'ORDER_PLACED', 'market', order.orderer, {'order_id': order_id, 'time': time, 'code': order.code, 'price': transaction_price, 'quantity': transaction_quantity}),
            time
        )

        self.market.send_message(
            Message('AGENT', 'ORDER_FILLED', 'market', order.orderer, {'code': code, 'price': transaction_price, 'quantity': transaction_quantity}),
            time
        )

        self.market.send_message(
            Message('AGENT', 'ORDER_FINISHED', 'market', order.orderer, {'code': code, 'time': time}),
            time
        )


            
    def cancel_order(self):
        pass

    def get_order(self, order_id):
        return self.orders[order_id]

    def get_stats(self):
        return self.stats
    
    def add_order(self, order_id, bid_or_ask, price, quantity):
        for i, pvo in enumerate(self.bids):
            if price == pvo[0]:
                fit_index = i
                break

        if bid_or_ask == 'BID':
            self.bids[fit_index][1] += quantity
            self.bids[fit_index][2].append([order_id, quantity])
            if fit_index > self.best_bid_index:
                self.best_bid_index = fit_index
        elif bid_or_ask == 'ASK':
            self.asks[fit_index][1] += quantity
            self.asks[fit_index][2].append([order_id, quantity])
            if fit_index < self.best_ask_index:
                self.best_ask_index = fit_index
        

    def match_order(self, bid_or_ask, price, quantity):
        transaction_quantity = 0
        transaction_amount = 0
        while quantity > 0:
            if bid_or_ask == 'BID' and price >= self.asks[self.best_ask_index][0]:
                pvo = self.asks
                price_index = self.best_ask_index
            elif bid_or_ask == 'ASK' and price <= self.bids[self.best_bid_index][0]:
                pvo = self.bids
                price_index = self.best_bid_index
            else:
                break

            while len(pvo[price_index][2]) != 0:
                matched_order_id = pvo[price_index][2][0][0]
                matched_order_quantity = pvo[price_index][2][0][1]

                if quantity >= matched_order_quantity:
                    matched_quantity = matched_order_quantity
                elif quantity < matched_order_quantity:
                    matched_quantity = matched_order_quantity
                
                self.fill_order(matched_order_id, pvo[price_index][0], matched_quantity)
                transaction_quantity += matched_quantity
                matched_order_quantity -= matched_quantity      
                pvo[price_index][1] -= matched_quantity
                quantity -= matched_quantity

                if matched_order_quantity == 0:
                    # remove the empty order
                    pvo[price_index][2].pop(0)

                if quantity == 0:
                    break
            
            transaction_amount += transaction_quantity * pvo[price_index][0]
            
            # check if best bid/ask needs to be updated
            if pvo[price_index][1] == 0:
                if bid_or_ask == 'BID' and price_index + 1 < len(pvo):
                    self.best_ask_index += 1
                elif bid_or_ask == 'ASK' and price_index - 1 >= 0:
                    self.best_bid_index -= 1

        return transaction_quantity, transaction_amount


    def daily_summarize(self):
        # clear unfinished orders
        for order_id, order in self.orders:
            # notify the orderer the invalidation of the unfinished order
            if order.state == False:
                self.market.send_message(
                    Message('AGENT', 'ORDER_INVALIDED', 'market', order.orderer, (order_id, order)),
                    self.market.get_time()
                )

        daily_summarization = {
            'date': self.market.get_time().date().isoformat(),
            'order': self.orders,
            'bid': self.bids,
            'ask': self.asks,
            'stats': self.stats
        }

        self.history.append(daily_summarization)
        self.reset()
        return daily_summarization['stats']

    def fill_order(self, order_id, price, quantity):
        # we don't update stats here, since one transection will be executed by two orders

        time = self.market.get_time()
        if price <= 0:
            raise Exception("Invalid value: negative price")
        elif quantity <= 0:
            raise Exception("Invalid value: negative quantity")

        target_order = self.get_order(order_id)
        target_order.transactions.append([time, price, quantity])

        # send message of transactions
        self.market.send_message(
            Message('AGENT', 'ORDER_FILLED', 'market', target_order.order.orderer, {'code': target_order.code, 'price': price, 'quantity': quantity}),
            time
        )

        # check if this order is finished
        total_quantity = 0
        for item in target_order.transactions:
            total_quantity += item[2]
        if total_quantity == target_order.order.quantity:
            target_order.state = True
            # send message of finishing order
            self.market.send_message(
                Message('AGENT', 'ORDER_FINISHED', 'market', target_order.order.orderer, {'code': target_order.code, 'time': time}),
                time
            )

    def update_stats(self, **name_val):
        for col, val in name_val:
            if col not in self.stats.keys():
                raise Exception('Invalid name')
            elif col in ['open', 'high', 'low', 'close']:
                self.stats[col] = val
            else:
                self.stats[col] += val

    def _generate_order_id(self):
        self.num_of_order += 1
        return f"{self.code}_{self.num_of_order:05}"

    def determine_tick_size(self, price):
        if price < 10:
            return 0.01
        elif price < 50:
            return 0.05
        elif price < 100:
            return 0.1
        elif price < 500:
            return 0.5
        elif price < 1000:
            return 1
        else:
            return 5