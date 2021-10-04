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
    core = Core(config, market_type="call")
    orderbooks, agent_manager = core.run(num_simulation = 1, num_of_timesteps = num_of_timesteps)
    write_records(orderbooks, agent_manager, result_dir)
    cost_time = str(timedelta(seconds = perf_counter() - start_time))
    print(f"Run {num_of_timesteps} in {cost_time}.")
    

if __name__ == '__main__':
    config_path = Path("config/scaling.json")
    config = json.loads(config_path.read_text())
    pre_value = [config['Market']['Securities']['TSMC']['value']]
    pre_price = [config['Market']['Securities']['TSMC']['value']]
    for i in range(249):
        pre_value.append(pre_value[-1] + round(random.gauss(0, 0.5), 1))
        pre_price.append(pre_price[-1] + round(random.gauss(0, 1), 1))
    pre_volume = [int(random.gauss(100, 10)*10) for i in range(249)]

    config['Market']['Securities']['TSMC']['value'] = pre_value
    config['Market']['Securities']['TSMC']['price'] = pre_price
    config['Market']['Securities']['TSMC']['volume'] = pre_volume
    experiment_name = 'call'
    result_dir = Path("result") / experiment_name
    simulate(result_dir / 'test', 2500, config, random_seed=9528)
