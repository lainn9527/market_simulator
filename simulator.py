import argparse
from market import Market
from zero_intelligence_agent import ZeroIntelligenceAgent
from core import Core

if __name__ == '__main__':
    market = Market({'TSMC': 300})
    agents = [ZeroIntelligenceAgent('zero_intelligence', start_cash = 10000000)]
    core = Core(market, agents)
    core.run()