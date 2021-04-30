import json
from pathlib import Path
from dataclasses import dataclass
from order import Order
from typing import List, Dict
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

def write_records(orderbooks: Dict, all_agents: Dict, output_dir: Path):
    output_dir = Path('./result/')
    if not output_dir.exists():
        output_dir.mkdir()
    # OHLCVM
    orderbook_stats = {}
    for code, orderbook in orderbooks.items():
        stats = {key: [] for key in orderbook.steps_record[0].keys()}
        for record in orderbook.steps_record:
            for k, v in record.items():
                stats[k].append(v)
        orderbook_stats[code] = stats
    
    for code, stats in orderbook_stats.items():
        file_path = output_dir / f"{code}.json"
        with open(file_path, 'w') as fp:
            json.dump(stats, fp, indent=4)
    
    agents_stats = {}
    for _type, agents in all_agents.items():
        stats = {key: [] for key in agents.holdings.keys()}
        for agent in agents:
            pass

    
    
