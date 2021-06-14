import argparse
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from core.core import Core
from core.market import Market
from core.order_book import OrderBook
from collections import defaultdict
from core.utils import OrderRecord, TransactionRecord
from core.order import LimitOrder

class TestCase:
    def test_place_bid_orders(self):
        config_path = Path("config/test_config.json")
        config = json.loads(config_path.read_text())

        price_list = [10.1, 10.15, 10.2, 10.1, 10]
        quantity_list =  [1, 2, 3, 1, 1]
        order_list = [{'code': 'TSMC', 'bid_or_ask': 'BID', 'price': price, 'quantity': quantity} for price, quantity in zip(price_list, quantity_list)]
        config['Agent']['TestAgent'][0]['order_list'] = order_list

        core = Core(config)
        orderbooks, agent_manager = core.run(num_simulation = 1, num_of_timesteps = 5)
        tsmc: OrderBook = orderbooks['TSMC']
        bid_orders = defaultdict(list)
        num_of_orders = 0
        cost = 0
        for price, num in zip(price_list, quantity_list):
            cost += price * num * 100
            num_of_orders += 1
            bid_orders[price].append(f"TSMC_{num_of_orders:05}")
        cost = cost * (1.002)
        agent = agent_manager.agents['te_0']

        assert tsmc.bids_price == [10.2, 10.15, 10.1, 10]
        assert tsmc.bids_volume == {10.2: 3, 10.15: 2, 10.1: 2, 10: 1}
        assert tsmc.bids_orders == bid_orders
        assert round(agent.reserved_cash) == round(cost)
        assert round(agent.cash) == round(10000 - cost)
    
    def test_place_ask_orders(self):
        config_path = Path("test_config.json")
        config = json.loads(config_path.read_text())

        price_list = [9.1, 9.15, 9.2, 9.1]
        quantity_list =  [1, 2, 3, 1]
        order_list = [{'code': 'TSMC', 'bid_or_ask': 'ASK', 'price': price, 'quantity': quantity} for price, quantity in zip(price_list, quantity_list)]
        config['Agent']['TestAgent'][0]['order_list'] = order_list

        core = Core(config)
        orderbooks, agent_manager = core.run(num_simulation = 1, num_of_timesteps = 10)
        tsmc: OrderBook = orderbooks['TSMC']

        ask_orders = defaultdict(list)
        num_of_orders = 0
        quantity = 0
        for price, num in zip(price_list, quantity_list):
            quantity += num
            num_of_orders += 1
            ask_orders[price].append(f"TSMC_{num_of_orders:05}")

        agent_id, agent = list(agent_manager.agents.items())[0]

        assert tsmc.asks_price == [9.1, 9.15, 9.2, 10]
        assert tsmc.asks_volume == {9.1: 2, 9.15: 2, 9.2: 3}
        assert tsmc.asks_orders == ask_orders
        assert agent.holdings['TSMC'] == 100 - quantity
        assert agent.reserved_holdings['TSMC'] == quantity


    def test_match_order(self):
        config_path = Path("config/test_config.json")
        config = json.loads(config_path.read_text())
        cash = config['Agent']['Global']['cash']
        holdings = config['Agent']['Global']['securities']['TSMC']
        transaction_rate = config['Market']['Structure']['transaction_rate']
        big_agent_name = config['Agent']['TestAgent'][0]['name']
        ask_agent_name = config['Agent']['TestAgent'][1]['name']
        value = 10

        bid_price_list = [10]
        bid_quantity_list =  [2]
        bid_order_list = [{'code': 'TSMC', 'bid_or_ask': 'BID', 'price': bid_price_list[i], 'quantity': bid_quantity_list[i], 'time': i} for i in range(len(bid_price_list))]

        time = len(bid_order_list)
        ask_price_list = [10]
        ask_quantity_list =  [3]
        ask_order_list = [{'code': 'TSMC', 'bid_or_ask': 'ASK', 'price': ask_price_list[i], 'quantity': ask_quantity_list[i], 'time': time+i} for i in range(len(ask_price_list))]

        config['Agent']['TestAgent'][0]['order_list'] = bid_order_list
        config['Agent']['TestAgent'][1]['order_list'] = ask_order_list

        core = Core(config)
        orderbooks, agent_manager = core.run(num_simulation = 1, num_of_timesteps = 3)
        tsmc: OrderBook = orderbooks['TSMC']

        bid_agent_id = agent_manager.group_agent[big_agent_name][0]
        bid_agent = agent_manager.agents[bid_agent_id]
        ask_agent_id = agent_manager.group_agent[ask_agent_name][0]
        ask_agent = agent_manager.agents[ask_agent_id]

        bid_order_id = bid_agent.orders_history['TSMC'][0]
        ask_order_id = ask_agent.orders['TSMC'][0]
        filled_price = bid_price_list[0]
        filled_quantity = bid_quantity_list[0]
        filled_amount = filled_price * filled_quantity * 100

        assert round(bid_agent.cash, 2) == round(cash - filled_amount*(1+transaction_rate), 2)
        assert bid_agent.holdings['TSMC'] == holdings + filled_quantity
        assert bid_agent.orders['TSMC'] == []
        assert bid_agent.orders_history['TSMC'] == [bid_order_id]

        assert round(ask_agent.cash, 2) == round(cash + filled_amount*(1-transaction_rate), 2)
        assert ask_agent.holdings['TSMC'] == holdings - filled_quantity - ask_agent.reserved_holdings['TSMC']
        assert ask_agent.orders['TSMC'] == [ask_order_id]
        assert ask_agent.orders_history['TSMC'] == []


        bid_order = tsmc.orders[bid_order_id]
        transaction = TransactionRecord(0, filled_price, filled_quantity)
        bid_order_record = OrderRecord(order = bid_order,
                                       placed_time = 0,
                                       finished_time = 0,
                                       transactions = [transaction],
                                       filled_quantity = filled_quantity,
                                       filled_amount = filled_amount,
                                       cancellation = False,
                                       transaction_cost=filled_amount*transaction_rate)
        ask_order = tsmc.orders[ask_order_id]
        ask_order_record = OrderRecord(order = ask_order,
                                       placed_time = 0,
                                       finished_time = None,
                                       transactions = [transaction],
                                       filled_quantity = filled_quantity,
                                       filled_amount = filled_amount,
                                       cancellation = False,
                                       transaction_cost=filled_amount*transaction_rate)

        assert tsmc.bids_price == []
        assert tsmc.asks_price == [9.1]
        assert tsmc.asks_volume == {9.1 : 1}
        assert tsmc.asks_orders == {9.1 : [ask_order_id]}
        assert tsmc.orders == {bid_order_id: bid_order, ask_order_id: ask_order}

    def test_match_orders(self):
        config_path = Path("config/test_config.json")
        config = json.loads(config_path.read_text())
        cash = 100000
        holdings = 100
        value = 10

        bid_price_list = [10.1, 10]
        bid_quantity_list =  [2, 4]
        bid_order_list = [{'code': 'TSMC', 'bid_or_ask': 'BID', 'price': price, 'quantity': quantity} for price, quantity in zip(bid_price_list, bid_quantity_list)]

        ask_price_list = [9.1, 10, 11, 9.5]
        ask_quantity_list =  [3, 0, 2, 2]
        ask_order_list = [{'code': 'TSMC', 'bid_or_ask': 'ASK', 'price': price, 'quantity': quantity} for price, quantity in zip(ask_price_list, ask_quantity_list)]

        config['Agent']['TestAgent'][0]['order_list'] = bid_order_list
        config['Agent']['TestAgent'][1]['order_list'] = ask_order_list

        core = Core(config)
        orderbooks, agent_manager = core.run(num_simulation = 1, num_of_timesteps = 5)
        tsmc: OrderBook = orderbooks['TSMC']

        bid_agent_id, bid_agent = list(agent_manager.agents.items())[0]
        ask_agent_id, ask_agent = list(agent_manager.agents.items())[1]
        bid_order_id = ['TSMC_00001', 'TSMC_00003']
        ask_order_id = ['TSMC_00002', 'TSMC_00004', 'TSMC_00005']

        # bid wealth
        last_filled_price = 10
        bid_cash = cash - 100 * (10.1 * 2 + 9.1 * 1 + 10 * 3 )
        bid_holdings = holdings + 5
        bid_reserved_cash = 100 * 10 * 1
        bid_wealth = bid_cash + bid_holdings * last_filled_price * 100 + bid_reserved_cash
        bid_orders = ['TSMC_00003']

        ask_cash = cash + 100 * (10.1 * 2 + 9.1 * 1 + 10 * 2 )
        ask_holdings = holdings - 7
        ask_reserved_holdings = 2
        ask_wealth = ask_cash + (ask_holdings + ask_reserved_holdings) * last_filled_price * 100
        ask_orders = ['TSMC_00004']


        assert bid_agent.cash == bid_cash
        assert bid_agent.holdings['TSMC'] == bid_holdings
        assert bid_agent.reserved_cash == bid_reserved_cash
        assert bid_agent.wealth == bid_wealth
        assert bid_agent.orders['TSMC'] == bid_orders

        assert ask_agent.cash == ask_cash
        assert ask_agent.holdings['TSMC'] == ask_holdings
        assert ask_agent.orders['TSMC'] == ask_orders
        assert ask_agent.wealth == ask_wealth
        assert ask_agent.orders['TSMC'] == ask_orders

        assert tsmc.bids_price == [10]
        assert tsmc.asks_price == [11]

    def test_cancel_order(self):
        config_path = Path("test_config.json")
        config = json.loads(config_path.read_text())

        price_list = [9.1, 9.15, 9.2, 9.1]
        quantity_list =  [1, 2, 3, 1]
        order_list = [{'code': 'TSMC', 'bid_or_ask': 'ASK', 'price': price, 'quantity': quantity} for price, quantity in zip(price_list, quantity_list)]
        config['Agent']['TestAgent'][0]['order_list'] = order_list

        core = Core(config)
        orderbooks, agent_manager = core.run(num_simulation = 1, num_of_timesteps = 202)
        tsmc: OrderBook = orderbooks['TSMC']
        agent_id, agent = list(agent_manager.agents.items())[0]

        assert agent.cash == 100000 * 1.01 * 1.01
        assert agent.holdings['TSMC'] == 100

if __name__ == '__main__':
    testcase = TestCase()
    testcase.test_match_order()