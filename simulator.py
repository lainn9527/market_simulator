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
    core = Core(config)
    orderbooks, agent_manager = core.run(num_simulation = 1, num_of_timesteps = num_of_timesteps)
    write_records(orderbooks, agent_manager, result_dir)
    cost_time = str(timedelta(seconds = perf_counter() - start_time))
    print(f"Run {num_of_timesteps} in {cost_time}.")
    

if __name__ == '__main__':
    config_path = Path("config/rule.json")
    config = json.loads(config_path.read_text())
    experiment_name = 'thesis_range5'
    result_dir = Path("simulation_result") / experiment_name
    simulate(result_dir, 4050, config, random_seed=9528)