import argparse
import datetime
from market import Market
from zero_intelligence_agent import ZeroIntelligenceAgent
from core import Core
from datetime import timedelta

class TestCase:
    def test_init(self, core):
        core.real_start_time = datetime.now()
        core.simulated_time = datetime.fromisoformat('2021-03-22 08:30:00.000')

    def test_basic_mechanism(self):
        market = Market({'TSMC': 300})
        agents = [ZeroIntelligenceAgent('zero_intelligence', start_cash = 10000000)]
        core = Core(market, agents)
        core.run()

if __name__ == '__main__':
    test = TestCase()
    test.test_basic_mechanism()