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
    config_path = Path("config/herd.json")
    config = json.loads(config_path.read_text())
    config['Core']['show_price'] = False
    experiment_name = 'dahoo'
    result_dir = Path("result") / experiment_name
    # simulate(result_dir / 'stock_numbe2', 5, config, random_seed=9528)
    scale = [0.01, 0.05] + [0.1 * i for i in range(1, 11)]
    cashes = [ int(i * 1.9e9) for i in scale]
    with Pool(8) as pool:
        for cash in cashes:
            tmp_config = deepcopy(config)
            tmp_config['Agent']['DahooAgent'][0]['cash'] = cash
            pool.apply_async(simulate, (result_dir / f"{int(cash//1.9e7)}_percent", 8100, tmp_config, 9527))
        pool.close()
        pool.join()


    # config0 = deepcopy(config)
    # config0['Agent']['DahooAgent'][0]['number'] = 0

    # config1 = deepcopy(config)
    # config1['Agent']['DahooAgent'][0]['cash'] = 1e7
    # config2 = deepcopy(config)
    # config2['Agent']['DahooAgent'][0]['cash'] = 2e7
    # config3 = deepcopy(config)
    # config3['Agent']['DahooAgent'][0]['cash'] = 4e7
    # config4 = deepcopy(config)
    # config4['Agent']['DahooAgent'][0]['cash'] = 5e7
    # config5 = deepcopy(config)
    # config5['Agent']['DahooAgent'][0]['number'] = 2
    # config5['Agent']['DahooAgent'][0]['cash'] = 1e7
    # config6 = deepcopy(config)
    # config6['Agent']['DahooAgent'][0]['number'] = 4
    # config6['Agent']['DahooAgent'][0]['cash'] = 1e7
    # config7 = deepcopy(config)
    # config7['Agent']['DahooAgent'][0]['cash'] = 1e8

    # names = ['base-no_dahoo', '1e7', '2e7', '4e7', '5e7', '2_1e7', '4_1e7', '1e8']
    # experiments = [
    #     (result_dir / names[0], 16200, config0,),
    #     (result_dir / names[1], 16200, config1,),
    #     (result_dir / names[2], 16200, config2,),
    #     (result_dir / names[3], 16200, config3,),
    #     (result_dir / names[4], 16200, config4,),
    #     (result_dir / names[5], 16200, config5,),
    #     (result_dir / names[6], 16200, config6,),
    #     (result_dir / names[7], 16200, config7,)
    # ]
    # with Pool(8) as pool:
    #     for args in experiments:
    #         pool.apply_async(simulate, args)
    #     pool.close()
    #     pool.join()
