import json
from pathlib import Path
from dataclasses import dataclass
from order import Order
from typing import List, Dict
from agent_manager import AgentManager
from collections import defaultdict
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
    output_dir = Path('./result/')
    if not output_dir.exists():
        output_dir.mkdir()
    # OHLCVM
    for code, orderbook in orderbooks.items():
        stats = defaultdict(list)
        for record in orderbook.steps_record:
            for k, v in record.items():
                stats[k].append(v)
        
        file_path = output_dir / f"{code}.json"
        with open(file_path, 'w') as fp:
            json.dump(stats, fp, indent=4)
    

    
    agents_stats = defaultdict(list)
    for record in agent_manager.step_records:
        for group_name, holdings in record.items():
            for code, value in holdings.items():
                agents_stats[f"{group_name}_{code}"].append(value)
        
    
    file_path = output_dir / "agent.json"
    with open(file_path, 'w') as fp:
        json.dump(agents_stats, fp, indent=4)

    
    
