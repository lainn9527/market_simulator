import argparse
from collections import defaultdict
import numpy as np
import random
import json
from time import perf_counter
from datetime import timedelta
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from multiprocessing import Pool, Process
from core.core import Core
from core.utils import write_records

def parse_args(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_name", type=str, default=config['Core']['result_name'])
    parser.add_argument("--timestep", type=int, default = config['Core']['timestep'])
    parser.add_argument("--num_zi", type=int, default=config['Agent']['ZeroIntelligenceAgent'][0]['number'])
    parser.add_argument("--num_dh", type=int, default=config['Agent']['DahooAgent'][0]['number'])
    parser.add_argument("--cash_dh", type=int, default=config['Agent']['DahooAgent'][0]['cash'])
    parser.add_argument("--show_price", action = 'store_true', default = False)
    args = parser.parse_args()

    config['Agent']['ZeroIntelligenceAgent'][0]['number'] = args.num_zi
    # print(config['Agent']['ZeroIntelligenceAgent'][0]['number'])
    config['Agent']['DahooAgent'][0]['number'] = args.num_dh
    config['Agent']['DahooAgent'][0]['cash'] = args.cash_dh
    config['Core']['timestep'] = args.timestep
    config['Core']['show_price'] = args.show_price
    return args

def simulate(result_dir: Path, timestep, config, random_seed = 9527):
    # args = parse_args(config)
    if not result_dir.exists():
        result_dir.mkdir(parents = True)
    with open(result_dir / 'config.json', 'w') as fp:
        json.dump(config, fp, indent=4)

    np.random.seed(random_seed)
    random.seed(random_seed)
    num_of_timesteps = timestep
    start_time = perf_counter()
    core = Core(config)
    orderbooks, agent_manager = core.run(num_simulation = 1, num_of_timesteps = num_of_timesteps)
    write_records(orderbooks, agent_manager, result_dir)
    cost_time = str(timedelta(seconds = perf_counter() - start_time))
    print(f"Run {num_of_timesteps} in {cost_time}.")
    

if __name__ == '__main__':
    config_path = Path("config/legacy/herd.json")
    config = json.loads(config_path.read_text())

    experiment_name = 'scaling'
    result_dir = Path("result") / experiment_name
    simulate(result_dir, 4050, config, random_seed=9528)