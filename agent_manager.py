import agent
import random
from typing import Dict, List
from collections import defaultdict
class AgentManager:
    def __init__(self, core, config: Dict):
        self.config = config
        self.global_config = self.config.pop("Global")
        self.agents = {}
        self.group_agent = {}
        self.core = core
        self.securities_list = []
        self.current_record = {}
        self.step_records = []
    
    def start(self, securities_list):
        self.securities_list += securities_list
        self.build_agents()
        for agent in self.agents.values():
            agent.start(self.core)

    def build_agents(self):
        for _type, groups in self.config.items():
            if len(groups) == 0:
                continue
            eval(f"self.build_{self.type_abbreviation(_type)}(groups)")


    def step(self):
        for agent in self.agents.values():
            agent.step()
        self.update_record()

    def receive_message(self, message):
        if message.postcode == 'AGENT':
            self.agents[message.receiver].receive_message(message)
        elif message.postcode == 'ALL_AGENTS':
            for agent in self.agents.values():
                agent.receive_message(message)
    
    def update_record(self):
        record = {}
        for group_name, agents in self.group_agent.items():
            holdings = defaultdict(float)
            for agent_id in agents:
                agent = self.agents[agent_id]
                holdings['CASH'] += agent.cash
                for code in agent.holdings.keys():
                    holdings[code] += agent.holdings[code]
            record[group_name] = {code: round(value/len(agents), 2) for code, value in holdings.items()}
        self.step_records.append(record)


    def build_zi(self, groups):
        group_conuter = 1
        for group in groups:
            group_name = f"zi_{group_conuter}"
            self.group_agent[group_name] = []
            securities = {code: 0 for code in self.securities_list}
            holdings = self.global_config['securities'] if 'securities' not in group.keys() else group['securities']
            securities.update(holdings)

            for i in range(group['number']):
                new_agent = agent.ZeroIntelligenceAgent(start_cash = self.global_config['cash'],
                                                        start_securities = securities,
                                                        bid_side = group['bid_side'],
                                                        range_of_price = group['range_of_price'],
                                                        range_of_quantity = group['range_of_quantity'])
                self.agents[new_agent.unique_id] = new_agent
                self.group_agent[group_name].append(new_agent.unique_id)
    
    def build_ch(self, groups):
        group_conuter = 1
        for group in groups:
            group_name = f"ch_{group_conuter}"
            self.group_agent[group_name] = []
            securities = {code: 0 for code in self.securities_list}
            holdings = self.global_config['securities'] if 'securities' not in group.keys() else group['securities']
            securities.update(holdings)

            for i in range(group['number']):
                group['risk_preference']
                new_agent = agent.ChartistAgent(start_cash = self.global_config['cash'],
                                                start_securities = securities,
                                                risk_preference = group['risk_preference'])
                self.agents[new_agent.unique_id] = new_agent
                self.group_agent[group_name].append(new_agent.unique_id)
            
    def type_abbreviation(self, _type):
        if _type == "ZeroIntelligenceAgent":
            return "zi"
        elif _type == "ChartistAgent":
            return "ch"
        elif _type == "FundamentalistAgent":
            return "fd"