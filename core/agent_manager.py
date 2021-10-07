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
        for agent in self.agents.values():
            agent.start(self.core)


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

    def parallel_env_step(self, actions):
        agent_ids = list(actions.keys())
        random.shuffle(agent_ids)
        for agent_id in agent_ids:
            self.agents[agent_id].step(actions[agent_id])

    def multi_env_step(self, actions):
        agent_ids = list(self.agents.keys())
        random.shuffle(agent_ids)
        for agent_id in agent_ids:
            if isinstance(self.agents[agent_id], agent.RLAgent):
                self.agents[agent_id].step(actions[agent_id])
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

    def get_init_states(self, group_name, number, cash, holdings, alpha):
        agent_ids = [f"{group_name}_{i}" for i in range(number)]
        agent_cashes = [ max(int(cash * number * pow(rank+1, -(1/alpha))), 10000) for rank in range(number)]
        agent_holdings = [{code: max(int(num * number * pow(rank+1, -(1/alpha))), 1) for code, num in holdings.items()} for rank in range(number)] 
        agent_average_costs = [random.gauss(mu = 100, sigma = 10) for _ in range(number)]
        risk_preferences = [random.gauss(mu = self.global_config['risk_preference_mean'], sigma = self.global_config['risk_preference_var']) for i in range(number)]

        if len(risk_preferences) > 1:
            min_risk, max_risk = min(risk_preferences), max(risk_preferences)
            risk_preferences = [ (risk_preference - min_risk) / (max_risk - min_risk) for risk_preference in risk_preferences]
        random.shuffle(agent_cashes)
        random.shuffle(agent_holdings)
        return agent_ids, agent_cashes, agent_holdings, agent_average_costs, risk_preferences

    def get_equal_init_states(self, group_name, number, cash, holdings):
        agent_ids = [f"{group_name}_{i}" for i in range(number)]
        agent_cashes = [cash for _ in range(number)]
        agent_holdings = [{code: num for code, num in holdings.items()} for _ in range(number)] 
        agent_average_costs = [random.gauss(mu = 100, sigma = 10) for _ in range(number)]
        risk_preferences = [random.gauss(mu = self.global_config['risk_preference_mean'], sigma = self.global_config['risk_preference_var']) for i in range(number)]
        if len(risk_preferences) > 1:
            min_risk, max_risk = min(risk_preferences), max(risk_preferences)
            risk_preferences = [ (risk_preference - min_risk) / (max_risk - min_risk) for risk_preference in risk_preferences]

        return agent_ids, agent_cashes, agent_holdings, agent_average_costs, risk_preferences
        
    def build_agents(self):
        '''
        The initial cash and securities are followed Pareto Law, also known as 80/20 rule.
        The risk preference follows Gaussian distribution but now it only affect the trend and mean revert agents.
        '''
        # reset number of scaling agent
        agent.ScalingAgent.reset_agent_number()

        for agent_type, groups in self.config.items():
            if len(groups) == 0:
                continue
            for config in groups:
                self.build_group_agent(agent_type, config)
        

    def build_group_agent(self, agent_type, config):
        # global configs
        n_agents = config['number']
        if n_agents == 0:
            return

        short_type = self.type_abbreviation(agent_type)
        group_name = f"{config['name']}_{n_agents}" if 'name' in config.keys() else f"{short_type}_{n_agents}"
        if group_name in self.group_agent.keys():
            raise Exception("Duplicate group name")

        original_cash = config['cash'] if 'cash' in config.keys() else self.global_config['cash']
        original_holdings = config['securities'] if 'securities' in config.keys() else self.global_config['securities']
        alpha = self.global_config['alpha']
        agent_ids, agent_cashes, agent_holdings, agent_average_costs, risk_preferences = self.get_init_states(group_name, n_agents, original_cash, original_holdings, alpha)
        # agent_ids, agent_cashes, agent_holdings, agent_average_costs, risk_preferences = self.get_equal_init_states(group_name, n_agents, original_cash, original_holdings)
        if agent_type == "ZeroIntelligenceAgent":
            range_of_price = config['range_of_price']
            range_of_quantity = config['range_of_quantity']
            bid_side = config['bid_side']
            agents = self.build_zi_agents(agent_ids, agent_cashes, agent_holdings, agent_average_costs, risk_preferences, range_of_price, range_of_quantity, bid_side)

        elif agent_type == "ParallelAgent":
            range_of_price = config['range_of_price']
            range_of_quantity = config['range_of_quantity']
            agents = self.build_pr_agents(agent_ids, agent_cashes, agent_holdings, agent_average_costs, risk_preferences, range_of_price, range_of_quantity)

        elif agent_type == "RLAgent":
            agents = self.build_rl_agents(agent_ids, agent_cashes, agent_holdings, agent_average_costs, risk_preferences)

        elif agent_type == "FundamentalistAgent":
            securities_value = {code: self.core.get_value(code) for code in self.securities_list}
            agents = self.build_fu_agent(agent_ids, agent_cashes, agent_holdings, agent_average_costs, risk_preferences, securities_value)

        elif agent_type == "ScalingAgent":
            range_of_price = config['range_of_price']
            range_of_quantity = config['range_of_quantity']
            group_type = config['type']
            agents = self.build_sc_agents(agent_ids, agent_cashes, agent_holdings, agent_average_costs, risk_preferences, range_of_price, range_of_quantity, group_type)
        # elif agent_type == "TrendAgent":
        #     return self.build_tr_agent(config, agent_cashes, agent_holdings, agent_average_costs, risk_preferences)
        # elif agent_type == "MeanRevertAgent":
        #     return self.build_mr_agent(config, agent_cashes, agent_holdings, agent_average_costs, risk_preferences)
        # elif agent_type == "TestAgent":
        #     return self.build_te_agent(config, agent_cashes, agent_holdings, agent_average_costs, risk_preferences)
        # elif agent_type == "RandomAgent":
        #     return self.build_ra_agent(config, agent_cashes, agent_holdings, agent_average_costs, risk_preferences)
        # elif agent_type == "DahooAgent":
        #     return self.build_dh_agent(config, agent_cashes, agent_holdings, agent_average_costs, risk_preferences)
        else:
            raise Exception(f"No {agent_type} agent")
        
        self.group_agent[group_name] = agent_ids
        self.initial_state[group_name] = {'cash': agent_cashes, 'security': agent_holdings}
        
        self.agents.update({agent_id: agent for agent_id, agent in zip(agent_ids, agents)})

    def add_rl_agent(self, config):
        rl_agent = agent.RLAgent(start_cash = config['cash'], start_securities = config['securities'], _id = config['id'])
        rl_agent.start(self.core)
        rl_agent.wealth = rl_agent.initial_wealth
        self.agents[rl_agent.unique_id] = rl_agent
        self.group_agent[config['name']] = [rl_agent.unique_id]

    def build_rl_agents(self, agent_ids, agent_cashes, agent_holdings, agent_average_costs, risk_preferences):
        agents = []
        n_agents = len(agent_ids)
        for i in range(n_agents):
            new_agent = agent.RLAgent(_id = agent_ids[i],
                                      start_cash = agent_cashes[i],
                                      start_securities = agent_holdings[i],
                                      average_cost = agent_average_costs[i],
                                      risk_preference = risk_preferences[i])
            agents.append(new_agent)
        
        return agents

    def build_sc_agents(self, agent_ids, agent_cashes, agent_holdings, agent_average_costs, risk_preferences, range_of_price, range_of_quantity, group_type):
        agents = []
        n_agents = len(agent_ids)
        for i in range(n_agents):
            new_agent = agent.ScalingAgent(_id = agent_ids[i],
                                           start_cash = agent_cashes[i], 
                                           start_securities = agent_holdings[i],
                                           average_cost= agent_average_costs[i],
                                           risk_preference = risk_preferences[i],
                                           range_of_price = range_of_price,
                                           range_of_quantity = range_of_quantity,
                                           group = group_type)
            agents.append(new_agent)
        
        return agents


    def build_zi_agents(self, agent_ids, agent_cashes, agent_holdings, agent_average_costs, risk_preferences, range_of_price, range_of_quantity, bid_side):
        agents = []
        n_agents = len(agent_ids)
        for i in range(n_agents):
            new_agent = agent.ZeroIntelligenceAgent(_id = agent_ids[i],
                                                    start_cash = agent_cashes[i],
                                                    start_securities = agent_holdings[i],
                                                    average_cost = agent_average_costs[i],
                                                    risk_preference = risk_preferences[i],
                                                    bid_side = bid_side,
                                                    range_of_price = range_of_price,
                                                    range_of_quantity = range_of_quantity)
            agents.append(new_agent)

        return agents

    def build_ra_agent(self, config):
        time_window = random.randint(10, config["range_of_time_window"])
        new_agent = agent.RandomAgent(_id = config['_id'],
                                      start_cash = config['cash'],
                                      start_securities = config['securities'],
                                      time_window = time_window,
                                      k = config['k'],
                                      mean = config['mean'])
        return new_agent

    def build_fu_agent(self, agent_ids, agent_cashes, agent_holdings, agent_average_costs, risk_preferences, securities_value):
        agents = []
        n_agents = len(agent_ids)
        for i in range(n_agents):
            new_agent = agent.FundamentalistAgent(_id = agent_ids[i],
                                                  start_cash = agent_cashes[i],
                                                  start_securities = agent_holdings[i],
                                                  average_cost = agent_average_costs[i],
                                                  risk_preference = risk_preferences[i],
                                                  securities_value = securities_value)
            agents.append(new_agent)
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


    def build_te_agent(self, config):
        securities = self.global_config['securities'].copy()
        new_agent = agent.TestAgent(_id = config['_id'],
                                    start_cash = config['cash'],
                                    start_securities = securities,
                                    order_list = config['order_list'])
        return new_agent


    def build_pr_agents(self, agent_ids, agent_cashes, agent_holdings, agent_average_costs, risk_preferences, range_of_price, range_of_quantity):
        agents = []
        n_agents = len(agent_ids)
        for i in range(n_agents):
            new_agent = agent.ParallelAgent(_id = agent_ids[i],
                                            start_cash = agent_cashes[i], 
                                            start_securities = agent_holdings[i],
                                            average_cost= agent_average_costs[i],
                                            risk_preference = risk_preferences[i],
                                            range_of_price = range_of_price,
                                            range_of_quantity = range_of_quantity)
            agents.append(new_agent)
        
        return agents

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
        elif _type == "ParallelAgent":
            return "pr"
        elif _type == "RLAgent":
            return "rl"
        elif _type == "ScalingAgent":
            return "sc"
        else:
            raise Exception(f"No {_type} agent")