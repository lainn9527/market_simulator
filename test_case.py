import argparse
from market import Market
from zero_intelligence_agent import ZeroIntelligenceAgent
from broker_agent import BrokerAgent
from core import Core
from datetime import datetime, timedelta

class TestCase:
    def test_init(self, core):
        core.real_start_time = datetime.now()
        core.simulated_time = datetime.fromisoformat('2021-03-22 08:30:00.000')

    def test_basic_mechanism(self):
        market = Market({'TSMC': 300})
        agents = [ZeroIntelligenceAgent(start_cash = 10000000) for i in range(100)] + [BrokerAgent('TSMC')]
        core = Core(market, agents)
        
        self.test_init(core)
        core.run(time_scale=0.1)

if __name__ == '__main__':
    test = TestCase()
    test.test_basic_mechanism()