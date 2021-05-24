import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict

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
    cancellation: bool

        

def write_records(orderbooks: Dict, agent_manager: AgentManager, output_dir: Path):
    if not output_dir.exists():
        output_dir.mkdir()
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
        for agent_id in agent_ids:
            initial_wealths.append(agent_manager.agents[agent_id].initial_wealth)
            returns.append((agent_manager.agents[agent_id].wealth - agent_manager.agents[agent_id].initial_wealth) / agent_manager.agents[agent_id].initial_wealth)
        agents_stats[group_name]['initial_wealth_by_agent'] = initial_wealths
        agents_stats[group_name]['returns_by_agent'] = returns
        agents_stats[group_name]['returns_by_step'] = list(np.diff(np.log(agents_stats[group_name]['WEALTH'])))

    file_path = output_dir / "agent.json"
    with open(file_path, 'w') as fp:
        json.dump(agents_stats, fp, indent=4)

    
    


    

    