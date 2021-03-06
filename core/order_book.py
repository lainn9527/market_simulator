import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from typing import Any, List, Tuple, Dict

from .order import LimitOrder, MarketOrder
from .message import Message
from .utils import OrderRecord, TransactionRecord


class OrderBook:
    def __init__(self, market, code, value, dividend_ratio, dividend_ar, dividend_var, dividend_period):
        """
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
        """
        self.market = market
        self.code = code
        self.value = value
        self.tick_size = None
        self.dividend_ratio = dividend_ratio
        self.dividend_ar = dividend_ar
        self.dividend_var = dividend_var
        self.dividend_period = dividend_period

        self.orders = dict()
        self.current_orders = []
        self.bids_price = list()
        self.bids_volume = defaultdict(int)
        self.bids_orders = defaultdict(list)
        self.bids_sum = 0
        self.asks_price = list()
        self.asks_volume = defaultdict(int)
        self.asks_orders = defaultdict(list)
        self.asks_sum = 0
        self.price_volume = defaultdict(int)

        self.num_of_order = 0
        self.num_of_cancelled_order = 0
        self.current_record: Dict = defaultdict(float)  # OHLCVA
        self.steps_record: Dict[str, List] = dict()  # OHLCVA
        self.dividend_record = dict()

    def set_price(self):
        base_price = self.value
        self.tick_size = self.market.determine_tick_size(base_price)
        record_list = [
            "value",
            "open",
            "high",
            "low",
            "close",
            "average",
            "volume",
            "amount",
            "bid",
            "ask",
            "price_volume",
            "bid_five_price",
            "ask_five_price",
        ]
        self.steps_record.update({key: [] for key in record_list})
        self.update_record(**{"value": self.value, "price": base_price, "volume": 0, "amount": 0})

    def handle_limit_order(self, order) -> str:
        """
        Match the limit order and fill the matched order.
        Step:
            1. add the limit order to the orderbook
            2. notify the agent of order placement
            3. match the limit order with best price and fill the matched order
            4. after there is no match pair, fill the limit order
            5. update the bid/ask price
            6. update the statics of the orderbook
        For order in auction, matching in step 3 and 4 is needless. The matching flag will control it.
        """

        time = self.market.get_time()
        order_id = self._generate_order_id()
        order.order_id = order_id
        order.price = round(order.price, 2)
        self.orders[order_id] = OrderRecord(order=order, placed_time=time)

        self.current_orders.append(order_id)
        mes_to_orderer = Message(
            postcode="AGENT",
            subject="ORDER_PLACED",
            sender="market",
            receiver=order.orderer,
            content={"code": order.code, "order_id": order_id},
        )

        self.market.send_message(mes_to_orderer)

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
            price = round(transaction_amount / (transaction_quantity * self.market.stock_size), 2)
            self.fill_order(order_id, price, transaction_quantity)
            to_update = {
                "price": last_price,
                "volume": transaction_quantity,
                "amount": transaction_amount,
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
                matched_order_quantity = matched_order_record.order.quantity - matched_order_record.filled_quantity

                if remain_quantity >= matched_order_quantity:
                    matched_quantity = matched_order_quantity
                elif remain_quantity < matched_order_quantity:
                    matched_quantity = remain_quantity

                self.fill_order(matched_order_id, best_ask_price, matched_quantity)
                last_price = best_ask_price
                transaction_quantity += matched_quantity
                transaction_amount += matched_quantity * best_ask_price
                remain_quantity -= matched_quantity
                self.price_volume[best_ask_price] += matched_quantity

                self.asks_volume[best_ask_price] -= matched_quantity
                self.asks_sum -= matched_quantity
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
        self.bids_sum += quantity

    def handle_ask_order(self, order_id):
        order = self.orders[order_id].order

        remain_quantity = order.quantity
        transaction_amount = 0
        transaction_quantity = 0
        # match the order if bid price is the best and >= the best ask price
        if (len(self.asks_price) == 0 or order.price < self.asks_price[0]) and len(self.bids_price) > 0:
            transaction_quantity, transaction_amount, last_price = self.match_ask_order(order.price, order.quantity)

        if transaction_quantity > 0:
            # if there is any match, fill the limit order
            price = round(transaction_amount / (transaction_quantity * self.market.stock_size), 2)
            self.fill_order(order_id, price, transaction_quantity)
            to_update = {
                "price": last_price,
                "volume": transaction_quantity,
                "amount": transaction_amount,
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
                matched_order_quantity = matched_order_record.order.quantity - matched_order_record.filled_quantity
                if remain_quantity >= matched_order_quantity:
                    matched_quantity = matched_order_quantity
                elif remain_quantity < matched_order_quantity:
                    matched_quantity = remain_quantity

                self.fill_order(matched_order_id, best_bid_price, matched_quantity)
                last_price = best_bid_price
                transaction_quantity += matched_quantity
                transaction_amount += matched_quantity * best_bid_price
                remain_quantity -= matched_quantity
                self.price_volume[best_bid_price] += matched_quantity

                self.bids_volume[best_bid_price] -= matched_quantity
                self.bids_sum -= matched_quantity
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
        self.asks_sum += quantity



    def fill_order(self, order_id, price, quantity):
        # we don't update stats here, since one transection will be executed by two orders
        if price <= 0:
            raise Exception("Invalid value: negative price")
        elif quantity <= 0:
            raise Exception("Invalid value: negative quantity")

        time = self.market.get_time()
        order_record = self.get_order(order_id)
        order_record.transactions.append(TransactionRecord(time=time, price=price, quantity=quantity))
        order_record.filled_quantity += quantity
        order_record.filled_amount += round(price * quantity * self.market.stock_size, 2)
        order_record.transaction_cost += round(price * quantity * self.market.stock_size * self.market.transaction_rate, 2)
        # send message of transactions
        filled_msg_content = {
            "code": order_record.order.code,
            "order_id": order_id,
            "bid_or_ask": order_record.order.bid_or_ask,
            "price": price,
            "quantity": quantity,
            "transaction_cost": order_record.transaction_cost,
        }

        filled_msg = Message(
            postcode="AGENT",
            subject="ORDER_FILLED",
            sender="market",
            receiver=order_record.order.orderer,
            content=filled_msg_content,
        )

        self.market.send_message(filled_msg)
        if order_record.filled_quantity == order_record.order.quantity:
            order_record.finished_time = time
            finished_msg = Message(
                postcode="AGENT",
                subject="ORDER_FINISHED",
                sender="market",
                receiver=order_record.order.orderer,
                content={"code": order_record.order.code, "order_id": order_record.order.order_id},
            )
            self.market.send_message(finished_msg)
            self.current_orders.remove(order_id)

        self.check_spread()

    def cancel_order(self, order_id):
        order_record = self.orders[order_id]
        order = order_record.order
        if order_record.finished_time is not None:
            # the order is finished before and we can't cancel it
            failed_msg = Message(
                postcode="AGENT",
                subject="ORDER_CANCEL_FAILED",
                sender="market",
                receiver=order_record.order.orderer,
                content={"code": order_record.order.code, "order_id": order_record.order.order_id},
            )
            self.market.send_message(failed_msg)
            return

        elif order_record.cancellation == True:
            # double cancelled
            raise Exception

        price, unfilled_quantity = order.price, order.quantity - order_record.filled_quantity
        unfilled_amount = round(price * unfilled_quantity * self.market.stock_size * (1 + self.market.transaction_rate), 2)
        self.current_orders.remove(order_id)
        self.orders[order_id].cancellation = True

        if order.bid_or_ask == "BID":
            self.bids_orders[price].remove(order_id)
            self.bids_volume[price] -= unfilled_quantity
            self.bids_sum -= unfilled_quantity
            if self.bids_volume[price] == 0:
                self.bids_price.remove(price)
        elif order.bid_or_ask == "ASK":
            self.asks_orders[price].remove(order_id)
            self.asks_volume[price] -= unfilled_quantity
            self.asks_sum -= unfilled_quantity
            if self.asks_volume[price] == 0:
                self.asks_price.remove(price)

        cancelled_content = {
            "code": order_record.order.code,
            "order_id": order_record.order.order_id,
            "refund_cash": unfilled_amount,
            "refund_security": unfilled_quantity,
        }
        cancelled_msg = Message(
            postcode="AGENT",
            subject="ORDER_CANCELLED",
            sender="market",
            receiver=order_record.order.orderer,
            content=cancelled_content,
        )
        self.market.send_message(cancelled_msg)
        # self.num_of_cancelled_order += 1

    def modify_order(self, modification_order):
        order_id = modification_order.order_id
        order_record = self.orders[order_id]
        filled_quantity = order_record.filled_quantity
        new_order = LimitOrder.from_modification_order(modification_order)
        new_order.quantity -= filled_quantity
        # calculate the refund
        original_cost = order_record.order.price * (order_record.order.quantity - filled_quantity)

        self.cancel_order(order_id)
        self.handle_limit_order(new_order)

        modified_content = {
            "code": new_order.code,
            "original_order_id": modification_order.order_id,
            "new_order_id": new_order.order_id,
        }
        modified_msg = Message(
            postcode="AGENT",
            subject="ORDER_MODIFIED",
            sender="market",
            receiver=modification_order.orderer,
            content=modified_content,
        )
        self.market.send_message(modified_msg)

    def update_record(self, **name_val):

        self.current_record["price"] = name_val["price"]
        self.current_record["open"] = name_val["price"] if "open" not in self.current_record.keys() else self.current_record["open"]
        self.current_record["high"] = max(self.current_record["high"], name_val["price"])
        self.current_record["low"] = name_val["price"] if "low" not in self.current_record.keys() else min(self.current_record["low"], name_val["price"])
        self.current_record["volume"] += name_val["volume"]
        self.current_record["amount"] += name_val["amount"]

    def step_summarize(self):
        self.current_record["close"] = self.current_record.pop("price")
        self.current_record["average"] = (
            round(self.current_record["amount"] / (100 * self.current_record["volume"]), 2)
            if self.current_record["amount"] != 0
            else self.current_record["close"]
        )
        self.current_record["amount"] = round(self.current_record["amount"], 2)
        self.current_record["bid"] = self.bids_price[0] if len(self.bids_price) > 0 else self.current_record["close"]
        self.current_record["ask"] = self.asks_price[0] if len(self.asks_price) > 0 else self.current_record["close"]
        self.current_record["bid_five_price"] = {price: self.bids_volume[price] for price in self.bids_price[:5]}
        self.current_record["ask_five_price"] = {price: self.asks_volume[price] for price in self.asks_price[:5]}
        self.current_record["price_volume"] = {price: volume for price, volume in self.price_volume.items()}

        for key in self.steps_record.keys():
            self.steps_record[key].append(self.current_record[key])

        self.current_record = defaultdict(float)
        updated_info = {
            "value": self.steps_record["value"][-1],
            "price": self.steps_record["close"][-1],
            "volume": 0,
            "amount": 0,
        }

        self.update_record(**updated_info)

    def clear_orders(self):
        for order_id in self.current_orders[:]:
            if self.market.get_time() - self.orders[order_id].placed_time >= self.market.clear_period:
                self.cancel_order(order_id)

    def check_spread(self):
        if len(self.bids_price) <= 1 or len(self.asks_price) <= 0:
            return

    def get_order(self, order_id):
        return self.orders[order_id]

    def adjust_value(self, v):
        self.current_record["value"] = v

    def _generate_order_id(self):
        self.num_of_order += 1
        return f"{self.code}_{self.num_of_order:05}"


class CallOrderBook(OrderBook):
    def __init__(self, market, code, dividend_ratio, dividend_ar, dividend_var, dividend_period, value):
        super().__init__(market, code, value, dividend_ratio, dividend_ar, dividend_var, dividend_period)

    def set_price(self):
        self.tick_size = self.market.determine_tick_size(self.steps_record["price"][-1])
        record_list = ['bid', 'ask', 'bid_five_price', 'ask_five_price', 'last_filled_bid_price', 'last_filled_ask_price']
        self.steps_record.update({key: [] for key in record_list})

        init_record = {
            "price": self.steps_record["price"][-1],
            "volume": self.steps_record["volume"][-1],
            "value": self.steps_record["value"][-1],
            "amount": self.steps_record["amount"][-1],
        }
        self.update_record(**init_record)

    def handle_bid_order(self, order_id):
        order = self.orders[order_id].order
        # update bid/ask if the limit order is partially filled and placed order
        self.quote_bid_order(order_id)

    def handle_ask_order(self, order_id):
        order = self.orders[order_id].order
        # update bid/ask if the limit order is partially filled and placed order
        self.quote_ask_order(order_id)

    def match_order(self):
        # construct accumulated volume of bids and asks???
        accum_bid_volume = self.bids_volume.copy()
        accum_ask_volume = self.asks_volume.copy()
        for i in range(1, len(self.bids_price)):
            accum_bid_volume[self.bids_price[i]] += accum_bid_volume[self.bids_price[i - 1]]
        for i in range(1, len(accum_ask_volume)):
            accum_ask_volume[self.asks_price[i]] += accum_ask_volume[self.asks_price[i - 1]]

        # for bid_price in self.bids_price
        max_match_volume = 0
        max_match_price = 0

        if np.random.binomial(n=1, p=0.5) == 1:
            bid_pointer = 0
            ask_pointer = len(self.asks_price) - 1
            while bid_pointer < len(self.bids_price) and ask_pointer >= 0:
                bid_price = self.bids_price[bid_pointer]
                ask_price = self.asks_price[ask_pointer]
                if bid_price < ask_price:
                    ask_pointer -= 1
                elif bid_price >= ask_price:
                    if max_match_volume < min(accum_bid_volume[bid_price], accum_ask_volume[ask_price]):
                        max_match_volume = min(accum_bid_volume[bid_price], accum_ask_volume[ask_price])
                        max_match_price = bid_price
                    bid_pointer += 1
                    if bid_price == ask_price:
                        ask_pointer -= 1
        else:
            bid_pointer = len(self.bids_price) - 1
            ask_pointer = 0
            while ask_pointer < len(self.asks_price) and bid_pointer >= 0:
                bid_price = self.bids_price[bid_pointer]
                ask_price = self.asks_price[ask_pointer]
                if ask_price > bid_price:
                    bid_pointer -= 1
                elif ask_price <= bid_price:
                    if max_match_volume < min(accum_bid_volume[bid_price], accum_ask_volume[ask_price]):
                        max_match_volume = min(accum_bid_volume[bid_price], accum_ask_volume[ask_price])
                        max_match_price = ask_price
                    ask_pointer += 1
                    if bid_price == ask_price:
                        bid_pointer -= 1

        if max_match_volume == 0:
            return 0

        return max_match_price, max_match_volume

    def fill_orders(self, match_price, match_volume):
        # fill the orders

        last_bid_remain = match_volume
        for bid_price in self.bids_price:
            if last_bid_remain <= 0 or bid_price < match_price:
                break
            for bid_order_id in self.bids_orders[bid_price]:
                bid_quantity = self.orders[bid_order_id].order.quantity
                self.fill_order(bid_order_id, match_price, min(bid_quantity, last_bid_remain))
                last_bid_remain -= bid_quantity
                last_filled_bid_price = bid_price
                if last_bid_remain <= 0:
                    break

        last_ask_remain = match_volume
        for ask_price in self.asks_price:
            if last_ask_remain <= 0 or ask_price > match_price:
                break
            for ask_order_id in self.asks_orders[ask_price]:
                ask_quantity = self.orders[ask_order_id].order.quantity
                self.fill_order(ask_order_id, match_price, min(ask_quantity, last_ask_remain))
                last_ask_remain -= ask_quantity
                last_filled_ask_price = ask_price
                if last_ask_remain <= 0:
                    break

        # update
        updated_info = {
            "price": match_price,
            "volume": match_volume,
            "amount": round(match_price * match_volume, 2),
            "bid": self.bids_sum,
            "ask": self.asks_sum,
            "last_filled_bid_price": last_filled_bid_price,
            "last_filled_ask_price": last_filled_ask_price,
        }

        return updated_info

    def handle_no_match(self):
        updated_info = {
            "volume": 0,
            "amount": 0,
            "bid": self.bids_sum,
            "ask": self.asks_sum,
        }

        return updated_info

    def clear_orders(self):
        for order_id in self.current_orders[:]:
            self.cancel_order(order_id)

    def update_record(self, **name_val):
        self.current_record.update(name_val)

    def step_summarize(self):
        self.current_record["bid_five_price"] = [price for price in self.bids_price[:5]]
        self.current_record["ask_five_price"] = [price for price in self.asks_price[:5]]
        for key in self.steps_record.keys():
            self.steps_record[key].append(self.current_record[key])

        # self.orders = dict()
        self.current_orders = []
        self.bids_price = list()
        self.bids_volume = defaultdict(int)
        self.bids_orders = defaultdict(list)
        self.bids_sum = 0
        self.asks_price = list()
        self.asks_volume = defaultdict(int)
        self.asks_orders = defaultdict(list)
        self.asks_sum = 0
        self.price_volume = defaultdict(int)

        self.num_of_order = 0
        self.num_of_cancelled_order = 0
