import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict

from numpy.lib.function_base import place

from .order import Order
from .agent_manager import AgentManager

@dataclass
class TransactionRecord:
    time: int
    price: float
    quantity: int


@dataclass
class OrderRecord:
    order: Order
    placed_time: int
    finished_time: int
    transactions: List[TransactionRecord]
    filled_quantity: int
    filled_amount: float
    transaction_cost: float
    cancellation: bool

        

def write_records(orderbooks: Dict, agent_manager: AgentManager, output_dir: Path):
    if not output_dir.exists():
        output_dir.mkdir()

    simulation_length = len(agent_manager.step_records)
    # OHLCVM
    for code, orderbook in orderbooks.items():
        file_path = output_dir / f"{code}.json"
        with open(file_path, 'w') as fp:
            json.dump(orderbook.steps_record, fp, indent=4)

    agents_stats = {group_name: defaultdict(list) for group_name in agent_manager.group_agent.keys()}
    for record in agent_manager.step_records:
        for group_name, holdings in record.items():
            for code, value in holdings.items():
                agents_stats[group_name][code].append(value)
            
        
    for group_name, agent_ids in agent_manager.group_agent.items():
        initial_wealths = []
        returns = []
        timestep_bid = [defaultdict(float) for _ in range(simulation_length)] # price: volume
        timestep_ask = [defaultdict(float) for _ in range(simulation_length)] # price: volume

        for agent_id in agent_ids:
            initial_wealths.append(agent_manager.agents[agent_id].initial_wealth)
            returns.append((agent_manager.agents[agent_id].wealth - agent_manager.agents[agent_id].initial_wealth) / agent_manager.agents[agent_id].initial_wealth)
            for order_id in agent_manager.agents[agent_id].orders_history['TSMC']:
                place_time = orderbooks['TSMC'].orders[order_id].placed_time
                order = price = orderbooks['TSMC'].orders[order_id].order
                if order.bid_or_ask == 'BID':
                    timestep_bid[place_time][order.price] += order.quantity
                elif order.bid_or_ask == 'ASK':
                    timestep_ask[place_time][order.price] += order.quantity

        agents_stats[group_name]['initial_wealth_by_agent'] = initial_wealths
        agents_stats[group_name]['initial_cash_by_agent'] = agent_manager.initial_state[group_name]['cash']
        agents_stats[group_name]['initial_security_by_agent'] = agent_manager.initial_state[group_name]['security']
        agents_stats[group_name]['returns_by_agent'] = returns
        # agents_stats[group_name]['returns_by_step'] = list(np.diff(np.log(agents_stats[group_name]['WEALTH'])))
        agents_stats[group_name]['timestep_bid'] = timestep_bid
        agents_stats[group_name]['timestep_ask'] = timestep_ask

    file_path = output_dir / "agent.json"
    with open(file_path, 'w') as fp:
        json.dump(agents_stats, fp, indent=4)

    
    
def write_multi_records(infos: Dict, output_dir: Path):
    output_dir.mkdir(exist_ok = True)
    length = len(infos['orderbooks'])
    for i in range(length):
        output_dir = output_dir / f'sim_{i+1}'
        output_dir.mkdir(exist_ok = True)
        write_records(infos['orderbooks'], infos['agent_manager'], output_dir)

    agent_states = infos['states']
    agent_states
    # rl_id = "RL"
    # rl_agent = agent_manager.agents[rl_id]
    # order_ids = rl_agent.orders['TSMC'] + rl_agent.orders_history['TSMC']

    # # output the order of rl agent
    # agent_orders = []
    # for order_id in order_ids:
    #     order_record = orderbooks['TSMC'].orders[order_id]
    #     order = order_record.order
    #     agent_orders.append({'time': int(order_record.placed_time), 'bid_or_ask': str(order.bid_or_ask), 'price': float(order.price), 'volume': int(order.quantity)})
    #     # order_price = order.price
    #     # state = states[order_record.placed_time]
    #     # close_price = state['price']['close']
    #     # slippage = close_price - order_price

    # file_path = output_dir / "rl.json"


    # with open(file_path, 'w') as fp:
    #     json.dump({'orders' :agent_orders, 'states': states}, fp, indent=4)
        