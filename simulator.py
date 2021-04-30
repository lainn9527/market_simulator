import argparse
import numpy as np
from market import Market
from agent.zero_intelligence_agent import ZeroIntelligenceAgent
from core import Core
from utils import write_records
if __name__ == '__main__':
    random_seed = 9527
    np.random.seed(random_seed)

    market = Market({'TSMC': 10})
    agents = {
        'ZeroIntelligenceAgent': [ZeroIntelligenceAgent(start_cash = 10000000, start_securities = {'TSMC': np.random.randint(1, 100)}) for i in range(1000)]
    }
    core = Core(market, agents)
    orderbooks, agents = core.run(num_simulation = 1, num_of_timesteps = 10000)
    write_records(orderbooks, agents, 'result/')