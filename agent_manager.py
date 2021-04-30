import agent
from typing import Dict, List
class AgentManager:
    def __init__(self, core, config: Dict):
        self.config = config
        self.global_config = self.config.pop("Global")
        self.agents = {}
        self.group_agent = {}
        self.core = core
    
    def start(self, securities):
        self.build_agents()
        for agent in self.agents.values():
            agent.start(self.core, securities)

    def build_agents(self):
        for _type, groups in self.config.items():
            if len(groups) == 0:
                continue
            eval(f"self.build_{self.type_abbreviation(_type)}(groups)")


    def step(self):
        for agent in self.agents.values():
            agent.step()

    def receive_message(self, message):
        if message.postcode == 'AGENT':
            self.agents[message.receiver].receive_message(message)
        elif message.postcode == 'ALL_AGENTS':
            for agent in self.agents.values():
                agent.receive_message(message)
    
    def build_zi(self, groups):
        group_conuter = 1
        for group in groups:
            group_name = f"zi_{group_conuter}"
            self.group_agent[group_name] = []
            for i in range(group['number']):
                new_agent = agent.ZeroIntelligenceAgent(start_cash = self.global_config['cash'],
                                                        start_securities = self.global_config['securities'],
                                                        bid_side = group['bid_side'],
                                                        range_of_price = group['range_of_price'],
                                                        range_of_quantity = group['range_of_quantity'])
                self.agents[new_agent.unique_id] = new_agent
                self.group_agent[group_name].append(new_agent.unique_id)
    
    def build_ch(self, groups):
        pass
            
    def type_abbreviation(self, _type):
        if _type == "ZeroIntelligenceAgent":
            return "zi"
        elif _type == "ChartistAgent":
            return "ch"
        elif _type == "FundamentalistAgent":
            return "fd"