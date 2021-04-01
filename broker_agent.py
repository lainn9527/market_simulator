import numpy as np
from agent import Agent

class BrokerAgent(Agent):
    num_of_agent = 0
    def __init__(self, _type, _id = None, start_cash = 1000000, security_unit = 1000):
        super().__init__(_type, _id, start_cash, security_unit)
        BrokerAgent.add_counter()