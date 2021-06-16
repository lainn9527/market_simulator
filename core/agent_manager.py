import random
from typing import Dict, List
from collections import defaultdict
from core import agent
from queue import Queue
class AgentManager:
    def __init__(self, core, config: Dict):
        self.config = config
        self.global_config = {}
        self.agents = {}
        self.agent_queue = Queue()
        self.group_agent = {}
        self.group_counter = {}
        self.group_stats = {}
        self.core = core
        self.securities_list = []
        self.current_record = {}
        self.initial_state = {}
        self.step_records = []


    def start(self, securities_list):
        self.global_config = self.config.pop("Global")
        self.securities_list += securities_list
        self.build_agents()


    # def step(self):
    #     agent_ids = list(self.agents.keys())
    #     random.shuffle(agent_ids)
    #     for agent_id in agent_ids:
    #         self.agents[agent_id].step()
    #     self.update_record()

    def step(self):
        if self.agent_queue.empty():
            agent_ids = list(self.agents.keys())
            random.shuffle(agent_ids)
            for agent_id in agent_ids:
                self.agent_queue.put(agent_id)
        while not self.core.queue_full() and not self.agent_queue.empty():
            agent_id = self.agent_queue.get()
            self.agents[agent_id].step()
        self.update_record() 

    def env_step(self, action, rl_agent_id):
        agent_ids = list(self.agents.keys())
        random.shuffle(agent_ids)
        for agent_id in agent_ids:
            if agent_id == rl_agent_id:
                status_code = self.agents[agent_id].step(action)
            else:
                self.agents[agent_id].step()
        self.update_record()

    def receive_message(self, message):
        if message.postcode == 'AGENT':
            self.agents[message.receiver].receive_message(message)
        elif message.postcode == 'ALL_AGENTS':
            for agent_id, agent in self.agents.items():
                agent.receive_message(message)
    
    def update_record(self):
        records = {}
        group_bids_volume = defaultdict(float)
        group_asks_volume = defaultdict(float)
        for group_name, agents in self.group_agent.items():
            record = defaultdict(float)
            for agent_id in agents:
                agent = self.agents[agent_id]
                record['cash'] += agent.cash
                for code in agent.holdings.keys():
                    record[code] += agent.holdings[code]
                record['wealth'] += agent.wealth
                record['average_cost'] += agent.average_cost
                record['group_bids_volume'] += agent.bids_volume
                record['group_asks_volume'] += agent.asks_volume
            records[group_name] = {code: round(value/len(agents), 2) for code, value in record.items() if code != 'group_bids_volume' or code != 'group_asks_volume'}
            records[group_name]['group_bids_volume'] = record['group_bids_volume']
            records[group_name]['group_asks_volume'] = record['group_asks_volume']
        self.step_records.append(records)

    def build_agents(self):
        '''
            The initial cash and securities are followed Pareto Law, also known as 80/20 rule.
            The risk preference follows Gaussian distribution but now it only affect the trend and mean revert agents.

        '''
        original_cash = self.global_config['cash']
        original_holdings = self.global_config['securities']
        # control the decreasing level
        alpha = self.global_config['alpha']
        for _type, groups in self.config.items():
            if len(groups) == 0:
                continue
            group_conuter = 1
            for config in groups:
                if config['number'] == 0:
                    continue
                original_cash = config['cash'] if 'cash' in config.keys() else self.global_config['cash']
                original_holdings = config['securities'] if 'securities' in config.keys() else self.global_config['securities']
                
                agent_cash = [ max(int(original_cash * config['number'] * pow(rank+1, -(1/alpha))), 10000) for rank in range(config['number'])]
                agent_holdings = [{code: max(int(num * config['number'] * pow(rank+1, -(1/alpha))), 1) for code, num in original_holdings.items()} for rank in range(config['number'])] 
                agent_average_cost = [random.gauss(mu = 100, sigma = 10) for _ in range(config['number'])]
                risk_preferences = [random.gauss(mu = self.global_config['risk_preference_mean'], sigma = self.global_config['risk_preference_var']) for i in range(config['number'])]

                if config['number'] > 1:
                    min_risk, max_risk = min(risk_preferences), max(risk_preferences)
                    risk_preferences = [ (risk_preference - min_risk) / (max_risk - min_risk) for risk_preference in risk_preferences]
                    
                short_type = self.type_abbreviation(_type)
                group_name = f"{config['name']}_{config['number']}"
                self.group_agent[group_name] = []
                self.group_counter[group_name] = 0
                for i in range(config['number']):
                    self.group_counter[group_name] += 1
                    config['_id'] = f"{group_name}_{self.group_counter[group_name]}"
                    config['cash'], config['securities'], config['risk_preference'], config['average_cost'] = agent_cash[i], agent_holdings[i], risk_preferences[i], agent_average_cost[i]
                    agent = eval(f"self.build_{short_type}_agent(config)")
                    agent.start(self.core)
                    self.agents[agent.unique_id] = agent
                    self.group_agent[group_name].append(agent.unique_id)
                
                self.initial_state[group_name] = {'cash': agent_cash, 'security': agent_holdings}
                    

    def add_rl_agent(self, config):
        rl_agent = agent.RLAgent(start_cash = config['cash'], start_securities = config['securities'], _id = config['id'])
        rl_agent.start(self.core)
        rl_agent.wealth = rl_agent.initial_wealth
        self.agents[rl_agent.unique_id] = rl_agent
        self.group_agent[config['name']] = [rl_agent.unique_id]


    def build_zi_agent(self, config):
        new_agent = agent.ZeroIntelligenceAgent(_id = config['_id'],
                                                start_cash = config['cash'],
                                                start_securities = config['securities'],
                                                bid_side = config['bid_side'],
                                                range_of_price = config['range_of_price'],
                                                range_of_quantity = config['range_of_quantity'])
        return new_agent

    def build_ra_agent(self, config):
        time_window = random.randint(10, config["range_of_time_window"])
        new_agent = agent.RandomAgent(_id = config['_id'],
                                      start_cash = config['cash'],
                                      start_securities = config['securities'],
                                      time_window = time_window,
                                      k = config['k'],
                                      mean = config['mean'])
        return new_agent

    def build_fu_agent(self, config):
        securities_vale = {code: self.core.get_value(code) for code in self.securities_list}
        new_agent = agent.FundamentalistAgent(_id = config['_id'],
                                              start_cash = config['cash'],
                                              start_securities = config['securities'],
                                              securities_value = securities_vale)
        return new_agent

    def build_tr_agent(self, config):
        strategy = config['strategy']
        time_window = random.randint(20, config["range_of_time_window"]) # > 20
        new_agent = agent.TrendAgent(_id = config['_id'],
                                     start_cash = config['cash'],
                                     start_securities = config['securities'],
                                     risk_preference = config['risk_preference'],
                                     strategy = strategy,
                                     time_window = time_window)
        return new_agent

    def build_mr_agent(self, config):
        strategy = config['strategy']
        time_window = random.randint(20, config["range_of_time_window"])
        new_agent = agent.MeanRevertAgent(_id = config['_id'], 
                                          start_cash = config['cash'],
                                          start_securities = config['securities'],
                                          risk_preference = config['risk_preference'],
                                          strategy = strategy,
                                          time_window = time_window)
        return new_agent

    def build_dh_agent(self, config):
        securities_vale = {code: self.core.get_value(code) for code in self.securities_list}
        new_agent = agent.DahooAgent(_id = config['_id'],
                                     start_cash = config['cash'],
                                     start_securities = config['securities'],
                                     securities_value = securities_vale)
        return new_agent
    def build_rl_agent(self):
        pass

    def build_te_agent(self, config):
        securities = self.global_config['securities'].copy()
        new_agent = agent.TestAgent(_id = config['_id'],
                                    start_cash = config['cash'],
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
        elif _type == "DahooAgent":
            return "dh"
        else:
            raise Exception(f"No {_type} agent")