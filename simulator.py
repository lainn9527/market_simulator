import argparse
from market import Market
from zero_intelligence_agent import ZeroIntelligenceAgent
from core import Core

if __name__ == '__main__':
    market = Market({'TSMC': 300})
    agents = {
        'ZeroIntelligenceAgent': [ZeroIntelligenceAgent(start_cash = 10000000) for i in range(100)]
    }
    core = Core(market, agents)
    core.run(num_simulation = 1, num_of_days = 1, time_scale = 0.1)