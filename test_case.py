import argparse
import numpy as np
from datetime import datetime, timedelta
from core import Core
from market import Market
from agent.test_agent import TestAgent
from order_book import OrderBook
class TestCase:
    def __init__(self):
        random_seed = 9527
        np.random.seed(random_seed)
        self.market = Market({'TSMC': 10})

    def test_place_bid_orders(self):
        price_list = [10.1, 10.15, 10.2, 10.1]
        quantity_list =  [1, 2, 3, 1]
        order_list = [{'code': 'TSMC', 'bid_or_ask': 'BID', 'price': price, 'quantity': quantity} for price, quantity in zip(price_list, quantity_list)]
        agents = {
            'TestAgent': [TestAgent(order_list = order_list)]
        }
        core = Core(self.market, agents)
        orderbooks, agents = core.run(num_simulation = 1, num_of_timesteps = 10)
        tsmc: OrderBook = orderbooks['TSMC']
        assert tsmc.bids_price == [10.2, 10.15, 10.1, 10]
        assert tsmc.bids_volume == {10.2: 3, 10.15: 2, 10.1: 2}
        print(tsmc.bids_orders)
    
    def test_place_ask_orders(self):
        price_list = [10.1, 10.15, 10.2, 10.1]
        quantity_list =  [1, 2, 3, 1]
        order_list = [{'code': 'TSMC', 'bid_or_ask': 'ASK', 'price': price, 'quantity': quantity} for price, quantity in zip(price_list, quantity_list)]
        agents = {
            'TestAgent': [TestAgent(order_list = order_list)]
        }
        core = Core(self.market, agents)
        orderbooks, agents = core.run(num_simulation = 1, num_of_timesteps = 10)
        tsmc: OrderBook = orderbooks['TSMC']
        assert tsmc.asks_price == [10, 10.1, 10.15, 10.2]
        assert tsmc.asks_volume == {10.2: 3, 10.15: 2, 10.1: 2}
        print(tsmc.asks_orders)

    def test_match_order(self):
        bid_order = {'code': 'TSMC', 'bid_or_ask': 'BID', 'price': 10.1, 'quantity': 1}
        ask_order = {'code': 'TSMC', 'bid_or_ask': 'ASK', 'price': 10.1, 'quantity': 1}
        bid_agent = TestAgent([bid_order])
        ask_agent = TestAgent([ask_order])
        agents = {'TestAgent': [bid_agent, ask_agent]}
        core = Core(self.market, agents)
        orderbooks, agents = core.run(num_simulation = 1, num_of_timesteps = 10)
        tsmc: OrderBook = orderbooks['TSMC']

    def test_match_orders(self):
        bid_price_list = [10.1, 10.15, 10.2, 10.1]
        bid_quantity_list =  [1, 2, 3, 1]
        ask_price_list = [10.1, 10.15, 10.2, 10.1]
        ask_quantity_list =  [1, 2, 3, 2]
        bid_order_list = [{'code': 'TSMC', 'bid_or_ask': 'BID', 'price': price, 'quantity': quantity} for price, quantity in zip(bid_price_list, bid_quantity_list)]
        ask_order_list = [{'code': 'TSMC', 'bid_or_ask': 'ASK', 'price': price, 'quantity': quantity} for price, quantity in zip(ask_price_list, ask_quantity_list)]
        bid_agent = TestAgent(bid_order_list)
        ask_agent = TestAgent(ask_order_list)
        agents = {'TestAgent': [bid_agent, ask_agent]}
        core = Core(self.market, agents)
        orderbooks, agents = core.run(num_simulation = 1, num_of_timesteps = 10)
        tsmc: OrderBook = orderbooks['TSMC']

if __name__ == '__main__':
    testcase = TestCase()
    testcase.test_match_orders()