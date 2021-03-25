import pandas as pd
from order import LimitOrder, MarketOrder
from dataclasses import dataclass
from typing import Any, List, Tuple

class OrderBook:
    def __init__(self, code):
        '''
        orders:
        {'Order ID':
            {
             'order': Order,
             'time': Time,
             'state': bool,
             'transactions': [],
             'modifications': [],
             'cancellation': bool
            }
        }
        code: code of security
        bids: [[price(descending), volumn, [[order_id, quantity]]]]
        asks: [[price(ascending), volumn, [[order_id, quantity]]]
        stats: {
            'amount': float,
            'volumn': int,
            'bid': int,
            'ask': int,
            'average': float,
            'high': float,
            'low': float,
        }
        '''
        self.orders = dict()
        self.transactions = list()
        self.code = code
        self.bids = list()
        self.asks = list()
        self.stats = dict()
    
    def match_order(self, order, time) -> Tuple[bool, List[LimitOrder]]:
        order_id = self._generate_order_id()
        self.orders[order_id] = {
            'order': order,
            'time': time,
            'state': False,
            'transactions': [],
            'modifications': [],
            'cancellation': False
        }
        remain_quantity = order.quantity
        
        transaction_amount = 0
        transaction_quantaty = 0
        matched_orders = []
        transactions = []
        while remain_quantity > 0:
            if order.bid_or_ask == 'BID' and order.price >= self.asks[0][0]:
                best_pvo = self.asks[0]
            elif order.bid_or_ask == 'ASK' and order.price <= self.bids[0][0]:
                best_pvo = self.bids[0]
            
            for matched_order_id, quantity in best_pvo[0][2]:
                if remain_quantity >= quantity:
                    matched_quantity = quantity
                elif remain_quantity < quantity:
                    matched_quantity = remain_quantity

                self.fill_order(matched_order_id, time, matched_quantity)
                transaction_quantaty += matched_quantity
                quantity -= matched_quantity      
                best_pvo[0][1] -= matched_quantity
                remain_quantity -= matched_quantity
                matched_orders.append((matched_order_id, matched_quantity))
                
                if remain_quantity == 0:
                    break
            
            transaction_amount += transaction_quantaty * best_pvo[0][0]
            # check if best bid/ask needs to be updated
            if best_pvo[0][1] == 0:
                if order.bid_or_ask == 'BID':
                    self.asks[0].pop(0)
                elif order.bid_or_ask == 'ASK':
                    self.bids[0].pop(0)
            else:
                # remove quantity == 0 order
                for i, n in enumerate(best_pvo[0][2]):
                    if n[1] != 0:
                        best_pvo[0][2] = best_pvo[0][2][i:]
                        break
                        



        if order.bid_or_ask == 'bid':
            pass
        elif order.bid_or_ask == 'ask':
            pass


    def add_order(self, order, time, matching):
        order_id = self._generate_order_id()
        self.orders[order_id] = {
            'order': order,
            'time': time,
            'state': False,
            'transactions': [],
            'modifications': [],
            'cancellation': False
        }
        if matching == False:
            # update bid & ask
            if order.bid_or_ask == 'bid':
                pass
            elif order.bid_or_ask == 'ask':
                pass
            return order_id
        else:
            pass
            
        
        
            

    def cancel_order(self):
        pass

    def get_order(self, order_id):
        return self.orders[order_id]

    def fill_order(self, order_id, time ,quantity):
        pass

    def _generate_order_id(self):
        pass
        