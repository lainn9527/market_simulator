
class Order:
    def __init__(self, orderer, code, order_type, bid_or_ask, quantity):
        self.code = code
        self.orderer = orderer
        self.order_type = order_type
        self.bid_or_ask = bid_or_ask
        self.quantity = quantity
        self.order_id = None
    
    

class LimitOrder(Order):
    def __init__(self, orderer, code, order_type, bid_or_ask, quantity, price):
        super().__init__(orderer, code, order_type, bid_or_ask, quantity)
        self.price = price
    
    @classmethod
    def from_market_order(cls, market_order, price):
        return cls(market_order.orderer, market_order.code, 'LIMIT', market_order.bid_or_ask, market_order.quantity, price)

    @classmethod
    def from_modification_order(cls, modification_order):
        return cls(modification_order.orderer, modification_order.code, 'LIMIT', modification_order.bid_or_ask, modification_order.quantity, modification_order.price)

    def __str__(self):
        return f"{self.order_id}, {self.bid_or_ask}, {self.price}, {self.quantity}"

    def __repr__(self):
        return f"{self.order_id}, {self.bid_or_ask}, {self.price}, {self.quantity}"


class MarketOrder(Order):
    def __init__(self, orderer, code, order_type, bid_or_ask, quantity):
        super().__init__(orderer, code, order_type, bid_or_ask, quantity)


class ModificationOrder(Order):
    def __init__(self, orderer, code, order_type, bid_or_ask, quantity, order_id, price = None):
        super().__init__(orderer, code, order_type, bid_or_ask, quantity)
        self.order_id = order_id
        self.price = price
        self.quantity = quantity

    @classmethod
    def from_limit_order(cls, limit_order, price, quantity, order_id):
        return cls(orderer = limit_order.orderer,
                   code = limit_order.code,
                   order_type = 'MODIFICATION',
                   bid_or_ask = limit_order.bid_or_ask,
                   quantity = quantity,
                   order_id = order_id,
                   price = price)
