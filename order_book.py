import random
from order import LimitOrder, MarketOrder
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, List, Tuple, Dict
from message import Message
from utils import OrderRecord, TransactionRecord


class OrderBook:
    def __init__(self, market, code, value, dividend_ratio, dividend_ar, dividend_var, dividend_period):
        '''
        history: [{
            'date': in isoformat (e.g. 2021-03-08),
            'order': orders,
            'bid': bids,
            'ask': asks,
            'stats': stats
        }]
        orders: {order_id: OrderRecord}
        code: code of security
        bids_price: [price]
        bids_volume: {price: volume}
        bids_orders: {price: [order_id]}

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
        self.tick_size = None
        self.history_order = dict()
        self.dividend_ratio = dividend_ratio
        self.dividend_ar = dividend_ar
        self.dividend_var = dividend_var
        self.dividend_period = dividend_period

        self.orders = dict()
        self.current_orders = []
        self.bids_price = list()
        self.bids_volume = defaultdict(int)
        self.bids_orders = defaultdict(list)
        self.asks_price = list()
        self.asks_volume = defaultdict(int)
        self.asks_orders = defaultdict(list)

        self.num_of_order = 0
        self.num_of_cancelled_order = 0
        self.current_record: Dict = defaultdict(float)  # OHLCVA
        self.steps_record: Dict[str, List] = dict()  # OHLCVA
        self.dividend_record = dict()

    def set_price(self):
        # base price for call auction in the open session
        # use the close price of previous day as the base price and if it's the first day, use the fundamental value instead
        base_price = self.value

        # initalize the valid price list
        tick_size = self.market.determine_tick_size(base_price)
        self.bids_price.append(base_price)
        self.asks_price.append(base_price)
        # point to the base price
        self.tick_size = tick_size

        record_list = ['open', 'high', 'low',
                       'close', 'average', 'volume', 'amount']
        self.steps_record.update({key: [] for key in record_list})
        self.update_record(**{'price': base_price, 'volume': 0, 'amount': 0})

    def handle_limit_order(self, order) -> str:
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
        order.order_id = order_id
        time = self.market.get_time()
        # check price
        order.price = round(order.price, 2)

        # add the order to the orderbook
        self.orders[order_id] = OrderRecord(order=order,
                                            placed_time=time,
                                            finished_time=None,
                                            transactions=[],
                                            filled_quantity=0,
                                            filled_amount=0,
                                            cancellation=False)
        self.current_orders.append(order_id)
        # send message to the orderer
        self.market.send_message(Message('AGENT', 'ORDER_PLACED', 'market', order.orderer, {
                                 'code': order.code, 'order_id': order_id}))

        if order.bid_or_ask == "BID":
            return self.handle_bid_order(order_id)
        elif order.bid_or_ask == "ASK":
            return self.handle_ask_order(order_id)

    def handle_bid_order(self, order_id):
        order = self.orders[order_id].order

        remain_quantity = order.quantity
        transaction_amount = 0
        transaction_quantity = 0

        # check liquidity
        # match the order if bid price is the best and >= the best ask price
        if (len(self.bids_price) == 0 or order.price > self.bids_price[0]) and len(self.asks_price) > 0:
            transaction_quantity, transaction_amount, last_price = self.match_bid_order(order.price, order.quantity)
        if transaction_quantity > 0:
            # if there is any match, fill the limit order
            price = round(transaction_amount/(transaction_quantity * self.market.stock_size), 2)
            self.fill_order(order_id, price, transaction_quantity)
            to_update = {
                'price': last_price,
                'volume': transaction_quantity,
                'amount': transaction_amount,
            }
            self.update_record(**to_update)
        # update bid/ask if the limit order is partially filled and placed order
        if transaction_quantity != order.quantity:
            self.quote_bid_order(order_id)

    def match_bid_order(self, price, quantity):
        remain_quantity = quantity
        transaction_quantity = 0
        transaction_amount = 0
        last_price = -1
        best_ask_price = self.asks_price[0]
        while remain_quantity > 0 and price >= best_ask_price:
            while len(self.asks_orders[best_ask_price]) != 0:
                matched_order_id = self.asks_orders[best_ask_price][0]
                matched_order_record = self.orders[matched_order_id]
                matched_order_quantity = matched_order_record.order.quantity - \
                    matched_order_record.filled_quantity

                if remain_quantity >= matched_order_quantity:
                    matched_quantity = matched_order_quantity
                elif remain_quantity < matched_order_quantity:
                    matched_quantity = remain_quantity

                self.fill_order(matched_order_id,
                                best_ask_price, matched_quantity)
                last_price = best_ask_price
                transaction_quantity += matched_quantity
                transaction_amount += matched_quantity * best_ask_price
                remain_quantity -= matched_quantity

                self.asks_volume[best_ask_price] -= matched_quantity
                if matched_order_record.filled_quantity == matched_order_record.order.quantity:
                    self.asks_orders[best_ask_price].pop(0)

                if remain_quantity == 0:
                    break

            # check if best bid/ask needs to be updated
            if self.asks_volume[best_ask_price] == 0:
                self.asks_price.pop(0)
                if len(self.asks_price) == 0:
                    break
            best_ask_price = self.asks_price[0]
        transaction_amount = self.market.stock_size * transaction_amount
        return transaction_quantity, transaction_amount, last_price

    def quote_bid_order(self, order_id):
        order_record = self.orders[order_id]
        price = order_record.order.price
        quantity = order_record.order.quantity - order_record.filled_quantity
        if len(self.bids_price) == 0 or price < self.bids_price[-1]:
            self.bids_price.append(price)
        else:
            for i in range(len(self.bids_price)):
                if price > self.bids_price[i]:
                    self.bids_price.insert(i, price)
                    break
                elif price == self.bids_price[i]:
                    break
        self.bids_orders[price].append(order_id)
        self.bids_volume[price] += quantity

    def handle_ask_order(self, order_id):
        order = self.orders[order_id].order

        remain_quantity = order.quantity
        transaction_amount = 0
        transaction_quantity = 0
        # match the order if bid price is the best and >= the best ask price
        if (len(self.asks_price) == 0 or order.price < self.asks_price[0]) and len(self.bids_price) > 0:
            transaction_quantity, transaction_amount, last_price = self.match_ask_order(
                order.price, order.quantity)

        if transaction_quantity > 0:
            # if there is any match, fill the limit order
            price = round(transaction_amount/(transaction_quantity * self.market.stock_size), 2)
            self.fill_order(order_id, price, transaction_quantity)
            to_update = {
                'price': last_price,
                'volume': transaction_quantity,
                'amount': transaction_amount,
            }
            self.update_record(**to_update)
        # update bid/ask if the limit order is partially filled and placed order
        if transaction_quantity != order.quantity:
            self.quote_ask_order(order_id)

    def match_ask_order(self, price, quantity):
        remain_quantity = quantity
        transaction_quantity = 0
        transaction_amount = 0
        last_price = -1
        best_bid_price = self.bids_price[0]
        while remain_quantity > 0 and price <= best_bid_price:
            while len(self.bids_orders[best_bid_price]) != 0:
                matched_order_id = self.bids_orders[best_bid_price][0]
                matched_order_record = self.orders[matched_order_id]
                matched_order_quantity = matched_order_record.order.quantity - \
                    matched_order_record.filled_quantity

                if remain_quantity >= matched_order_quantity:
                    matched_quantity = matched_order_quantity
                elif remain_quantity < matched_order_quantity:
                    matched_quantity = remain_quantity

                self.fill_order(matched_order_id,
                                best_bid_price, matched_quantity)
                last_price = best_bid_price
                transaction_quantity += matched_quantity
                transaction_amount += matched_quantity * best_bid_price
                remain_quantity -= matched_quantity

                self.bids_volume[best_bid_price] -= matched_quantity
                if matched_order_record.filled_quantity == matched_order_record.order.quantity:
                    self.bids_orders[best_bid_price].pop(0)

                if remain_quantity == 0:
                    break

            # check if best bid needs to be updated
            if self.bids_volume[best_bid_price] == 0:
                self.bids_price.pop(0)
                if len(self.bids_price) == 0:
                    break
            best_bid_price = self.bids_price[0]
        
        transaction_amount = self.market.stock_size * transaction_amount
        return transaction_quantity, transaction_amount, last_price

    def quote_ask_order(self, order_id):
        order_record = self.orders[order_id]
        price = order_record.order.price
        quantity = order_record.order.quantity - order_record.filled_quantity
        if len(self.asks_price) == 0 or price > self.asks_price[-1]:
            self.asks_price.append(price)
        else:
            for i in range(len(self.asks_price)):
                if price < self.asks_price[i]:
                    self.asks_price.insert(i, price)
                    break
                elif price == self.asks_price[i]:
                    break
        self.asks_orders[price].append(order_id)
        self.asks_volume[price] += quantity

    # def handle_market_order(self, order, time):
    #     time = self.market.get_time()

    #     # assume there is no liquidity issue
    #     # add the limit order in the best price in the market

    #     price = self.stats['up_limit'] if order.bid_or_ask == 'BID' else self.stats['low_limit']
    #     transaction_quantity, transaction_amount = self.match_order(order.bid_or_ask, price, order.quantity)
    #     transaction_price = round(transaction_amount / transaction_quantity, 2)
    #     if transaction_quantity != order.quantity:
    #         raise Exception("Liquidity error: The market order isn't finished.")

    #     order_id = self._generate_order_id()
    #     limit_order = LimitOrder.from_market_order(order, transaction_price)
    #     limit_order.order_id = order_id

    #     self.orders[order_id] = {
    #         'order': limit_order,
    #         'time': time,
    #         'state': True,
    #         'transactions': [[time, transaction_price, transaction_quantity]],
    #         'modifications': [],
    #         'cancellation': False
    #     }

    #     self.market.send_message(
    #         Message('AGENT', 'ORDER_PLACED', 'market', order.orderer, {'order': limit_order, 'time': time}),
    #         time
    #     )

    #     self.market.send_message(
    #         Message('AGENT', 'ORDER_FILLED', 'market', order.orderer, {'code': code, 'order_id': order_id, 'price': transaction_price, 'quantity': transaction_quantity}),
    #         time
    #     )

    #     self.market.send_message(
    #         Message('AGENT', 'ORDER_FINISHED', 'market', order.orderer, {'code': code, 'order_id': order_id, 'time': time}),
    #         time
    #     )

    def fill_order(self, order_id, price, quantity):
        # we don't update stats here, since one transection will be executed by two orders
        if price <= 0:
            raise Exception("Invalid value: negative price")
        elif quantity <= 0:
            raise Exception("Invalid value: negative quantity")

        time = self.market.get_time()
        order_record = self.get_order(order_id)
        order_record.transactions.append(TransactionRecord(
            time=time, price=price, quantity=quantity))
        order_record.filled_quantity += quantity
        order_record.filled_amount += price * quantity * self.market.stock_size
        # send message of transactions
        self.market.send_message(
            Message('AGENT', 'ORDER_FILLED', 'market', order_record.order.orderer, {
                    'code': order_record.order.code, 'order_id': order_id, 'bid_or_ask': order_record.order.bid_or_ask, 'price': price, 'quantity': quantity}),
        )

        # check if this order is finished
        if order_record.filled_quantity == order_record.order.quantity:
            order_record.finished_time = time
            # send message of finishing order
            self.market.send_message(
                Message('AGENT', 'ORDER_FINISHED', 'market', order_record.order.orderer, {
                        'code': order_record.order.code, 'order_id': order_record.order.order_id})
            )
            self.current_orders.remove(order_id)

        self.check_spread()

    def cancel_order(self, order_id):
        order_record = self.orders[order_id]
        order = order_record.order
        price, unfilled_quantity = order.price, order.quantity - order_record.filled_quantity
        unfilled_amount = price * unfilled_quantity * self.market.stock_size

        if order.bid_or_ask == 'BID':
            self.bids_orders[price].remove(order_id)
            self.bids_volume[price] -= unfilled_quantity
            if self.bids_volume[price] == 0:
                self.bids_price.remove(price)
        elif order.bid_or_ask == 'ASK':
            self.asks_orders[price].remove(order_id)
            self.asks_volume[price] -= unfilled_quantity
            if self.asks_volume[price] == 0:
                self.asks_price.remove(price)
        
        self.current_orders.remove(order_id)
        self.orders[order_id].cancellation = True
        
        self.market.send_message(
            Message('AGENT', 'ORDER_CANCELLED', 'market', order_record.order.orderer,
                    {'code': order_record.order.code, 'order_id': order_record.order.order_id, 'unfilled_amount': unfilled_amount, 'unfilled_quantity': unfilled_quantity})
        )
        # self.num_of_cancelled_order += 1

    def modify_order(self):
        pass

    def update_record(self, **name_val):
        # OHLCVA
        if 'open' not in self.current_record.keys():
            self.current_record['open'] = name_val['price']
        self.current_record['price'] = name_val['price']
        self.current_record['high'] = max(
            self.current_record['high'], name_val['price'])
        if 'low' not in self.current_record.keys():
            self.current_record['low'] = name_val['price']
        else:
            self.current_record['low'] = min(
                self.current_record['low'], name_val['price'])
        self.current_record['volume'] += name_val['volume']
        self.current_record['amount'] += name_val['amount']

    def step_summarize(self):
        self.current_record['close'] = self.current_record.pop('price')
        self.current_record['average'] = round(self.current_record['amount']/ (100*self.current_record['volume']), 2) if self.current_record['amount'] != 0 else self.current_record['close']
        self.current_record['amount'] = round(self.current_record['amount'], 2)

        for key in self.steps_record.keys():
            self.steps_record[key].append(self.current_record[key])

        self.current_record = defaultdict(float)
        self.update_record(
            **{'price': self.steps_record['close'][-1], 'volume': 0, 'amount': 0})

    def clear_orders(self):
        for order_id in self.current_orders[:]:
            if self.market.get_time() - self.orders[order_id].placed_time >= 100:
                self.cancel_order(order_id)
            
    def check_spread(self):
        if len(self.bids_price) <= 1 or len(self.asks_price) <= 0:
            return
        if self.bids_price[0] >= self.asks_price[0]:
            print("for break")

    def get_order(self, order_id):
        return self.orders[order_id]

    def _generate_order_id(self):
        self.num_of_order += 1
        return f"{self.code}_{self.num_of_order:05}"
