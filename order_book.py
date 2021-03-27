import pandas as pd
from order import LimitOrder, MarketOrder
from dataclasses import dataclass
from typing import Any, List, Tuple
from core import Message
class OrderBook:
    def __init__(self, market, code, value):
        '''
            history: [date, [orders]], past orders, sorted by day
            daily_info:
            [
                (
                    date,
                    {
                        'open': float,
                        'high': float,
                        'low': float,
                        'close': float
                    }
                )
            ]
            orders:
            {
                'Order ID':
                    {
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
            stats:
            {
                'amount': float,
                'volume': int,
                'bid': int,
                'ask': int,
                'average': float,
                'high': float,
                'low': float,
            }
        '''
        self.market = market
        self.code = code
        self.value = value

        self.history = dict()
        self.daily_info = []

        self.base_price = 0
        self.orders = dict()
        self.bids = list()
        self.asks = list()
        self.num_of_order = 0
        self.stats = {
            'amount': 0.0,
            'volume': 0,
            'bid': 0,
            'ask': 0,
            'average': 0.0,
            'high': 0,
            'low': 0,
        }
    
    def set_base_price(self):
        # base price for call auction in the open session
        # use the close price of previous day as the base price and if it's the first day, use the fundamental value instead

        if len(self.daily_info) == 0:
            self.base_price = self.value
        else:
            self.base_price = self.daily_info[-1][1]['close']
        return self.base_price


    def handle_limit_order(self, order, time, matching) -> str:
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
            Message('AGENT', 'ORDER_PLACED', 'market', order.orderer, {'price': order.price,'quantity': order.quantity}),
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
        self.stats['amount'] += transaction_amount
        self.stats['volume'] += transaction_quantity
        self.stats['average'] = round(self.stats['amount'] / (self.stats['volume'] + 1e-6), 2)
        self.stats['high'] = order.price if order.price > self.stats['high'] else self.stats['high']
        self.stats['low'] = order.price if order.price < self.stats['low'] else self.stats['low']
        if order.bid_or_ask == 'BID':
            if remain_quantity > 0:
                self.stats['bid'] += remain_quantity
            self.stats['ask'] -= transaction_quantity
        elif order.bid_or_ask == 'ASK':
            if remain_quantity > 0:
                self.stats['ask'] += remain_quantity
            self.stats['bid'] -= transaction_quantity
        
        return order_id

    def handle_market_order(self, order, time):
        # assume there is no liquidity issue
        # add the limit order in the best price in the market
        best_price = self.asks[0][0] if order.bid_or_ask == 'BID' else self.bids[0][0]
        limit_order_id = self.handle_limit_order(LimitOrder.from_market_order(order, best_price), time)

        # check if the order is finished        
        if self.get_order(limit_order_id).state == 0:
            raise Exception("The market order isn't finished")
        

    def cancel_order(self):
        pass

    def get_order(self, order_id):
        return self.orders[order_id]

    def fill_order(self, order_id, time, price, quantity):
        if price <= 0:
            raise Exception("Invalid value: negative price")
        elif quantity <= 0:
            raise Exception("Invalid value: negative quantity")

        target_order = self.get_order(order_id)
        target_order.transactions.append([time, price, quantity])

        # send message of transactions
        self.market.send_message(
            Message('AGENT', 'ORDER_FILLED', 'market', target_order.order.orderer, {'price': price, 'quantity': quantity}),
            time
        )

        # check if this order is finished
        total_quantity = 0
        total_amount = 0
        for item in target_order.transactions:
            total_quantity += item[2]
            total_amount += item[1] * item[2]
        if total_quantity == target_order.order.quantity:
            target_order.state = 1
            # send message of finishing order
            self.market.send_message(
                Message('AGENT', 'ORDER_FINISHED', 'market', target_order.order.orderer, {'price': round(total_amount/total_quantity, 2),'quantity': total_quantity}),
                time
            )

    def clear_orders(self):
        date = self.market.get_time().date()
        self.history[date] = self.orders()
        pass
    def _generate_order_id(self):
        self.num_of_order += 1
        return f"{self.code}_{self.num_of_order:05}"
        