import argparse
import numpy as np
import json
import random
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
        bid_agent_name = config['Agent']['TestAgent'][0]['name']
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

        bid_agent_id = agent_manager.group_agent[bid_agent_name][0]
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

    def test_hedge_order(self):
        config_path = Path("config/test_config.json")
        config = json.loads(config_path.read_text())
        cash = config['Agent']['Global']['cash']
        holdings = config['Agent']['Global']['securities']['TSMC']
        transaction_rate = config['Market']['Structure']['transaction_rate']

        bid_hedge_name = f"{config['Agent']['TestAgent'][0]['name']}_1"
        bid_hedge_order_list = [{'code': 'TSMC', 'bid_or_ask': 'BID', 'price': 10, 'quantity': 2, 'time': 0},
                                {'code': 'TSMC', 'bid_or_ask': 'BID', 'price': 9, 'quantity': 2, 'time': 1},
                                {'code': 'TSMC', 'bid_or_ask': 'BID', 'price': 4, 'quantity': 2, 'time': 2},
                                {'code': 'TSMC', 'bid_or_ask': 'ASK', 'price': 8, 'quantity': 5, 'time': 3},]
        time = len(bid_hedge_order_list)
        config['Agent']['TestAgent'][0]['order_list'] = bid_hedge_order_list
        bid_hedge_final_cash = round(cash - 4 * 2 * 100 * (1+transaction_rate) + 8 * 2 * 100 * (1-transaction_rate), 2)
        bid_hedge_final_holding = holdings - 5
        # The first two bid orders will be cancelled
        # So the final cash is ( origin - 4 * 2 * 100 * (1+0.002) ) and the final holdings is ( origin - 5)

        ask_hedge_name = f"{config['Agent']['TestAgent'][1]['name']}_1"
        ask_hedge_order_list = [{'code': 'TSMC', 'bid_or_ask': 'ASK', 'price': 7, 'quantity': 5, 'time': 4},
                                {'code': 'TSMC', 'bid_or_ask': 'BID', 'price': 12, 'quantity': 2, 'time': 5},]

        config['Agent']['TestAgent'][1]['order_list'] = ask_hedge_order_list
        ask_hedge_final_cash = round(cash - 8 * 2 * 100 * (1+transaction_rate), 2)
        ask_hedge_final_holding = holdings + 2

        # The only transaction is ask 1 at price 8. The order remain is bid 1 at price 13
        # So the final cash is ( origin + 8 * 1 * 100 * (1-0.002) - 13 * 1 * 100 * (1+0.002) ) and the final holding is ( origin - 1)


        core = Core(config)
        orderbooks, agent_manager = core.run(num_simulation = 1, num_of_timesteps = 8)
        tsmc: OrderBook = orderbooks['TSMC']

        bid_hedge_agent_id = agent_manager.group_agent[bid_hedge_name][0]
        bid_hedge_agent = agent_manager.agents[bid_hedge_agent_id]

        ask_hedge_agent_id = agent_manager.group_agent[ask_hedge_name][0]
        ask_hedge_agent = agent_manager.agents[ask_hedge_agent_id]

        bid_order_id = bid_hedge_agent.orders_history['TSMC'][0]
        # filled_price = bid_price_list[0]
        # filled_quantity = bid_quantity_list[0]
        # filled_amount = filled_price * filled_quantity * 100

        assert round(bid_hedge_agent.cash, 2) == bid_hedge_final_cash
        assert bid_hedge_agent.holdings['TSMC'] == bid_hedge_final_holding

        assert round(ask_hedge_agent.cash, 2) == ask_hedge_final_cash
        assert ask_hedge_agent.holdings['TSMC'] == ask_hedge_final_holding

    def test_call_market(self):
        config_path = Path("config/test_call_market.json")
        config = json.loads(config_path.read_text())
        cash = config['Agent']['Global']['cash']
        holdings = config['Agent']['Global']['securities']['TSMC']
        transaction_rate = config['Market']['Structure']['transaction_rate']
        config['Market']['Securities']['TSMC']['price'] = [random.gauss(100, 10) for i in range(100)]
        config['Market']['Securities']['TSMC']['volume'] = [int(random.gauss(100, 10)*10) for i in range(100)]
        config['Market']['Securities']['TSMC']['value'] = [random.gauss(100, 10) for i in range(100)]


        bid_agent_name = f"{config['Agent']['TestAgent'][0]['name']}_1"
        bid_agent_order_list = [{'code': 'TSMC', 'bid_or_ask': 'BID', 'price': 9, 'quantity': 2, 'time': 0},
                                {'code': 'TSMC', 'bid_or_ask': 'BID', 'price': 10, 'quantity': 2, 'time': 0},
                                {'code': 'TSMC', 'bid_or_ask': 'BID', 'price': 11, 'quantity': 2, 'time': 0},
                                {'code': 'TSMC', 'bid_or_ask': 'BID', 'price': 9, 'quantity': 2, 'time': 1},
                                {'code': 'TSMC', 'bid_or_ask': 'BID', 'price': 10, 'quantity': 2, 'time': 1},
                                {'code': 'TSMC', 'bid_or_ask': 'BID', 'price': 11, 'quantity': 2, 'time': 1},]
        time = len(bid_agent_order_list)
        config['Agent']['TestAgent'][0]['order_list'] = bid_agent_order_list
        bid_agent_final_cash = round(cash - 9 * 10 * 100 * (1+transaction_rate), 2)
        bid_agent_final_holding = holdings + 10
        # The first two bid orders will be cancelled
        # So the final cash is ( origin - 4 * 2 * 100 * (1+0.002) ) and the final holdings is ( origin - 5)

        ask_agent_name = f"{config['Agent']['TestAgent'][1]['name']}_1"
        ask_agent_order_list = [{'code': 'TSMC', 'bid_or_ask': 'ASK', 'price': 7, 'quantity': 5, 'time': 0},
                                {'code': 'TSMC', 'bid_or_ask': 'ASK', 'price': 12, 'quantity': 2, 'time': 0},
                                {'code': 'TSMC', 'bid_or_ask': 'ASK', 'price': 7, 'quantity': 5, 'time': 1},
                                {'code': 'TSMC', 'bid_or_ask': 'ASK', 'price': 12, 'quantity': 2, 'time': 1}]

        config['Agent']['TestAgent'][1]['order_list'] = ask_agent_order_list
        ask_agent_final_cash = round(cash + 9 * 10 * 100 * (1-transaction_rate), 2)
        ask_agent_final_holding = holdings - 10

        # The only transaction is ask 1 at price 8. The order remain is bid 1 at price 13
        # So the final cash is ( origin + 8 * 1 * 100 * (1-0.002) - 13 * 1 * 100 * (1+0.002) ) and the final holding is ( origin - 1)


        core = Core(config, market_type="call")
        orderbooks, agent_manager = core.run(num_simulation = 1, num_of_timesteps = 3)
        tsmc: OrderBook = orderbooks['TSMC']

        bid_agent_id = agent_manager.group_agent[bid_agent_name][0]
        bid_agent = agent_manager.agents[bid_agent_id]

        ask_agent_id = agent_manager.group_agent[ask_agent_name][0]
        ask_agent = agent_manager.agents[ask_agent_id]

        # bid_order_id = bid_agent.orders_history['TSMC'][0]
        # filled_price = bid_price_list[0]
        # filled_quantity = bid_quantity_list[0]
        # filled_amount = filled_price * filled_quantity * 100

        assert round(bid_agent.cash, 2) == bid_agent_final_cash
        assert bid_agent.holdings['TSMC'] == bid_agent_final_holding

        assert round(ask_agent.cash, 2) == ask_agent_final_cash
        assert ask_agent.holdings['TSMC'] == ask_agent_final_holding

        
        

if __name__ == '__main__':
    testcase = TestCase()
    testcase.test_call_market()