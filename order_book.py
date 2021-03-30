import pandas as pd
import datetime
from order import LimitOrder, MarketOrder
from dataclasses import dataclass
from typing import Any, List, Tuple
from core import Message
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
            bids: [[price(descending), volume, [[order_id, quantity]]]]
            asks: [[price(ascending), volume, [[order_id, quantity]]]
            stats: {
                'value: float
                'base': float,
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
        self.num_of_order = None
        self.stats = None
        self.reset()

    def reset(self):
        self.orders = dict()
        self.bids = list()
        self.asks = list()
        self.num_of_order = 0
        self.stats = {
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
    
    def set_base_price(self):
        # base price for call auction in the open session
        # use the close price of previous day as the base price and if it's the first day, use the fundamental value instead

        if len(self.history) == 0:
            self.stats['base'] = self.value
        else:
            self.stats['base'] = self.history[-1]['stats']['close']
        
        return self.stats['base']

    def handle_auction(self):
        time = self.market.get_time()
        # match price of the bid and ask with max volume
        if len(self.bids) == 0 or len(self.asks) == 0:
            print('No quote!')
            return
        if self.bids[0][0] < self.asks[0][0]:
            print('No match...')
            return

        # locate the best bid at asks
        best_bid = self.bids[0][0]
        match_index = -1
        for i, pvo in enumerate(self.asks):
            if best_bid == pvo[0]:
                match_index = i
                break
        if match_index == -1:
            raise Exception('Liquidity Error: No successive quotes')

        # construct accumulated volume of bids and asks
        accum_bid = [pvo[1] for pvo in self.bids]
        accum_ask = [pvo[1] for pvo in self.asks]
        for i in range(1, accum_bid):
            accum_bid[i] = accum_bid[i] + accum_bid[i-1]
        for i in range(1, accum_ask):
            accum_ask[i] = accum_ask[i] + accum_ask[i-1]
        
        # match the max volume
        i = 0
        j = match_index
        max_match_volume = 0
        match_bid_index = -1
        match_ask_index = -1
        while i < len(self.bids) and j >= 0:
            match = min(accum_bid[i], accum_ask[j])
            if match > max_match_volume:
                max_match_volume = match
                match_bid_index = i
                match_ask_index = j
            i += 1
            j -= 1
        
        match_price = self.bids[match_bid_index][0]

        # fill the orders
        for pvos in [self.bids[:match_bid_index+1], self.asks[:match_ask_index+1]]:
            remain_quantity = max_match_volume
            while remain_quantity > 0:
                pvo = pvos[0]
                if pvo[0] != match_price or remain_quantity == pvo[1]:
                    # fill all orders
                    remain_quantity -= pvo[1]
                    for order_id, quantity in pvo[2]:
                        self.fill_order(order_id, time, pvo[0], quantity)
                    # remove the pvo from the bids/asks
                    pvos.pop(0)
                else:
                    pvo[1] -= remain_quantity
                    while remain_quantity > pvo[2][0][1]:
                        self.fill_order(pvo[2][0][0], time, pvo[0], pvo[2][0][1])
                        remain_quantity -= pvo[2][0][1]
                        # remove order with zero quantity
                        pvo.pop(0)
                    self.fill_order(pvo[2][0][0], time, pvo[0], remain_quantity)
                    pvo[2][0][1] -= remain_quantity

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
            
        # match the limit order with best price
        while matching and remain_quantity > 0:
            # if there is no order, stop matching
            if order.bid_or_ask == 'BID' and len(self.asks) > 0 and order.price >= self.asks[0][0]:
                best_pvo = self.asks[0]
            elif order.bid_or_ask == 'ASK' and len(self.bids) > 0 and order.price <= self.bids[0][0]:
                best_pvo = self.bids[0]
            else:
                break
            
            # best_pvo: [price(descending), volume, [[order_id, quantity]] ]
            for matched_order_id, quantity in best_pvo[2]:
                if remain_quantity >= quantity:
                    matched_quantity = quantity
                elif remain_quantity < quantity:
                    matched_quantity = remain_quantity

                self.fill_order(matched_order_id, time, best_pvo[0], matched_quantity)
                transaction_quantity += matched_quantity
                quantity -= matched_quantity      
                best_pvo[1] -= matched_quantity
                remain_quantity -= matched_quantity
                
                if remain_quantity == 0:
                    break
            
            transaction_amount += transaction_quantity * best_pvo[0]
            # check if best bid/ask needs to be updated
            if best_pvo[1] == 0:
                if order.bid_or_ask == 'BID':
                    self.asks.pop(0)
                elif order.bid_or_ask == 'ASK':
                    self.bids.pop(0)
            else:
                # remove order with zero quantity (since remove item when iterating list may cause index error, we don't remove the order in the previous for-loop)
                for i, n in enumerate(best_pvo[2]):
                    if n[1] != 0:
                        best_pvo[2] = best_pvo[2][i:]
                        break
        
        # if there is any match, fill the limit order
        if transaction_quantity > 0:
            self.fill_order(order_id, time, round(transaction_amount/transaction_quantity, 2), transaction_quantity)

        # update bid/ask if the limit order is partially filled and placed order
        if remain_quantity > 0:
            if order.bid_or_ask == 'BID':
                for i, pvo in enumerate(self.bids):
                    if order.prive > pvo[0]:
                        self.bids.insert(i, [order.price, remain_quantity, [order_id, remain_quantity]])
                        break
                    elif order.price == pvo[0]:
                        pvo[1] += remain_quantity
                        pvo[2].append([order.price, remain_quantity, [order_id, remain_quantity]])
                        break
                    if i == len(self.bids) - 1:
                        self.bids.append([order.price, remain_quantity, [order_id, remain_quantity]])
            elif order.bid_or_ask == 'ASK':
                for i, pvo in enumerate(self.asks):
                    if order.prive < pvo[0]:
                        self.asks.insert(i, [order.price, remain_quantity, [order_id, remain_quantity]])
                        break
                    elif order.price == pvo[0]:
                        pvo[1] += remain_quantity
                        pvo[2].append([order.price, remain_quantity, [order_id, remain_quantity]])
                        break
                    if i == len(self.asks) - 1:
                        self.asks.append([order.price, remain_quantity, [order_id, remain_quantity]])
        
        # update the stats
        updated_info = {
            'amount': transaction_amount,
            'volume': transaction_quantity,
            'average': round(self.stats['amount'] / (self.stats['volume'] + 1e-6), 2),
            'high': order.price if order.price > self.stats['high'] else self.stats['high'],
            'low': order.price if order.price < self.stats['low'] else self.stats['low'],
            'bid': remain_quantity if order.bid_or_ask == 'BID' else (-1*transaction_quantity),
            'ask': remain_quantity if order.bid_or_ask == 'ASK' else (-1*transaction_quantity)
        }
        self.update_stats(**updated_info)
        
        return order_id
    
        
    def handle_market_order(self, order, time):
        time = self.market.get_time()
        
        # assume there is no liquidity issue
        # add the limit order in the best price in the market
        best_price = self.asks[0][0] if order.bid_or_ask == 'BID' else self.bids[0][0]
        limit_order_id = self.handle_limit_order(LimitOrder.from_market_order(order, best_price), time, True)

        # check if the order is finished        
        if self.get_order(limit_order_id).state == False:
            raise Exception("The market order isn't finished")
            
    def cancel_order(self):
        pass

    def get_order(self, order_id):
        return self.orders[order_id]

    def get_stats(self):
        return self.stats

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

    def fill_order(self, order_id, time, price, quantity):
        # we don't update stats here, since one transection will be executed by two orders

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
            target_order.state = 1
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
        