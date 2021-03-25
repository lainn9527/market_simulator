class Order:
    def __init__(self, orderer, code, order_type, bid_or_ask, quantity):
        self.code = code
        self.orderer = orderer
        self.order_type = order_type
        self.bid_or_ask = bid_or_ask
        self.quantity = quantity

class LimitOrder(Order):
    def __init__(self, orderer, code, order_type, bid_or_ask, quantity, price):
        super().__init__(orderer, code, order_type, bid_or_ask, quantity)
        self.price = price


class MarketOrder(Order):
    def __init__(self, orderer, code, order_type, bid_or_ask, quantity):
        super().__init__(orderer, code, order_type, bid_or_ask, quantity)


class ModificationOrder(Order):
    def __init__(self, orderer, code, order_type, bid_or_ask, quantity, order_id, price = None):
        super().__init__(orderer, code, order_type, bid_or_ask, quantity)
        self.order_id = order_id
        self.price = price
