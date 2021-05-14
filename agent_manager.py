import agent
import random
from typing import Dict, List
from collections import defaultdict
from multiprocessing import Pool

class AgentManager:
    def __init__(self, core, config: Dict):
        self.config = config
        self.global_config = {}
        self.agents = {}
        self.group_agent = {}
        self.core = core
        self.securities_list = []
        self.current_record = {}
        self.step_records = []

    def start(self, securities_list):
        self.global_config = self.config.pop("Global")
        self.securities_list += securities_list
        self.build_agents()
        for agent in self.agents.values():
            agent.start(self.core)

    def step(self):
        agent_ids = list(self.agents.keys())
        random.shuffle(agent_ids)
        for agent_id in agent_ids:
            self.agents[agent_id].step()
        self.update_record()

    def receive_message(self, message):
        if message.postcode == 'AGENT':
            self.agents[message.receiver].receive_message(message)
        elif message.postcode == 'ALL_AGENTS':
            for agent_id, agent in self.agents.items():
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
                holdings['WEALTH'] += agent.wealth
                # if agent.cash > self.global_config['cash'] * 2:
                    # print('first genius')
            record[group_name] = {code: round(value/len(agents), 2) for code, value in holdings.items()}
        
        self.step_records.append(record)

    def build_agents(self):
        # Pareto law
        original_cash = self.global_config['cash']
        original_holdings = self.global_config['securities']
        # control the decreasing level
        alpha = self.global_config['alpha']
        for _type, groups in self.config.items():
            if len(groups) == 0:
                continue
            group_conuter = 1
            for config in groups:
                # set up the global config
                agent_cash = [round(original_cash * config['number'] * pow(rank+1, -(1/alpha))) for rank in range(config['number'])]
                agent_holdings = [{code: round(num * config['number'] * pow(rank+1, -(1/alpha))) for code, num in original_holdings.items()} for rank in range(config['number'])] 
                risk_preferences = [random.gauss(mu = self.global_config['risk_preference_mean'], sigma = self.global_config['risk_preference_var']) for i in range(config['number'])]
                # scale to 0~1
                min_risk, max_risk = min(risk_preferences), max(risk_preferences)
                risk_preferences = [ (risk_preference - min_risk) / (max_risk - min_risk) for risk_preference in risk_preferences]

                short_type = self.type_abbreviation(_type)
                group_name = f"{short_type}_{group_conuter}" if 'name' not in config.keys() else config['name']
                self.group_agent[group_name] = []
                for i in range(config['number']):
                    config['cash'], config['securities'], config['risk_preference'] = agent_cash[i], agent_holdings[i], risk_preferences[i]
                    agent = eval(f"self.build_{short_type}_agent(config)")
                    self.agents[agent.unique_id] = agent
                    self.group_agent[group_name].append(agent.unique_id)

                    


    def build_zi_agent(self, config):
        new_agent = agent.ZeroIntelligenceAgent(start_cash = config['cash'],
                                                start_securities = config['securities'],
                                                bid_side = config['bid_side'],
                                                range_of_price = config['range_of_price'],
                                                range_of_quantity = config['range_of_quantity'])
        return new_agent

    def build_ra_agent(self, config):
        time_window = random.randint(10, config["range_of_time_window"])
        new_agent = agent.RandomAgent(start_cash = config['cash'],
                                        start_securities = config['securities'],
                                        time_window = time_window,
                                        k = config['k'],
                                        mean = config['mean'])
        return new_agent

    def build_fu_agent(self, config):
        securities_vale = {code: self.core.get_value(code) for code in self.securities_list}
        new_agent = agent.FundamentalistAgent(start_cash = config['cash'],
                                              start_securities = config['securities'],
                                              securities_value = securities_vale)
        return new_agent

    def build_tr_agent(self, config):
        strategy = config['strategy']
        time_window = random.randint(1, config["range_of_time_window"])
        strategy.update({"time_window": time_window})
        new_agent = agent.TrendAgent(start_cash = config['cash'],
                                     start_securities = config['securities'],
                                     risk_preference = config['risk_preference'],
                                     strategy = strategy)
        return new_agent

    def build_mr_agent(self, config):
        strategy = config['strategy']
        time_window = random.randint(1, config["range_of_time_window"])
        strategy.update({"time_window": time_window})
        new_agent = agent.MeanRevertAgent(start_cash = config['cash'],
                                          start_securities = config['securities'],
                                          risk_preference = config['risk_preference'],
                                          strategy = strategy)
        return new_agent

    def build_te_agent(self, config):
        securities = self.global_config['securities'].copy()
        new_agent = agent.TestAgent(start_cash = config['cash'],
                                    start_securities = securities,
                                    order_list = config['order_list'])
        return new_agent

    def type_abbreviation(self, _type):
        if _type == "ZeroIntelligenceAgent":
            return "zi"
        elif _type == "ChartistAgent":
            return "ch"
        elif _type == "FundamentalistAgent":
            return "fu"
        elif _type == "TrendAgent":
            return "tr"
        elif _type == "MeanRevertAgent":
            return "mr"
        elif _type == "TestAgent":
            return "te"
        elif _type == "RandomAgent":
            return "ra"
        else:
            raise Exception(f"No {_type} agent")