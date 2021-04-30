import argparse
import numpy as np
import json
from market import Market
from agent import ZeroIntelligenceAgent
from core import Core
from utils import write_records
from pathlib import Path


if __name__ == '__main__':
    config_path = Path("config.json")
    result_dir_path = Path("result/")
    random_seed = 9527
    np.random.seed(random_seed)


    config = json.loads(config_path.read_text())
    core = Core(config)
    orderbooks, agent_manager = core.run(num_simulation = 1, num_of_timesteps = 10000)
    write_records(orderbooks, agent_manager, result_dir_path)