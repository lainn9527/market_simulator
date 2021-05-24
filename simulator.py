import argparse
import numpy as np
import random
import json
from time import perf_counter
from datetime import timedelta
from pathlib import Path
from datetime import datetime

from core.core import Core
from core.utils import write_records
# config = {
#     "Market":{
#         "Structure": {
#             "interest_rate": 0.01,
#             "interest_period": 100,
#             "clear_period": 20
#         },
#         "Securities": {
#             "TSMC": {
#                 "value": 100,
#                 "dividend_ratio": 0.1,
#                 "dividend_ar": 0.95,
#                 "dividend_var": 0.0743,
#                 "dividend_period": 100
#             }
#         }
#     },

#     "Agent": {
#         "Global": {
#             "cash": 100000,
#             "securities": {
#                 "TSMC": 10
#             }
#         },
#         "RandomAgent": [
#             {
#                 "number": 10000,
#                 "range_of_time_window": 100, 
#                 "k": 3.85,
#                 "mean": 1.01
#             }
#         ]
#     }
# }
if __name__ == '__main__':
    config_path = Path("config/config_zi.json")
    result_dir = Path("result/zi_1000/")
    if not result_dir.exists():
        result_dir.mkdir()
    

    random_seed = 9527
    np.random.seed(random_seed)
    random.seed(random_seed)
    num_of_timesteps = 10000
    start_time = perf_counter()

    config = json.loads(config_path.read_text())
    core = Core(config)
    orderbooks, agent_manager = core.run(num_simulation = 1, num_of_timesteps = num_of_timesteps)
    write_records(orderbooks, agent_manager, result_dir)
    with open(result_dir / 'config.json', 'w') as fp:
        json.dump(config, fp)
    cost_time = str(timedelta(seconds = perf_counter() - start_time))
    print(f"Run {num_of_timesteps} in {cost_time}.")
    