import random
import numpy as np
import gym
import json
import torch
from core import agent
from collections import defaultdict
from typing import Dict, List
from gym import spaces
from gym.utils import seeding
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from core.core import Core
from .rl_agent import BaseAgent



class MultiTradingEnv:

    def __init__(self, config: Dict):
        self.config = config
        self.train = True
    
    def build_agents(self,
                     resume,
                     resume_model_dir,
                     lr,
                     device,
                     agent_config):

        # device = torch.device(train_config['device'])

        # build
        if resume:
            resume_config_path = resume_model_dir / 'config.json'
            resume_model_path = resume_model_dir / 'model.pkl'
            resume_config = json.loads(resume_config_path.read_text())
            agent_config = resume_config['Agent']['RLAgent']
        
        agents = []
        for config in agent_config:
            agents += self.build_agent(config)
        
        if resume:
            checkpoint = torch.load(resume_model_path)
            for i, agent in enumerate(agents):
                agent.rl.load_state_dict(checkpoint[f"base_{i}"])
            print(f"Resume {len(agents)} rl agents from {resume_model_path}.")
        else:
            print(f"Initiate {len(agents)} rl agents.")
        
        return agents

    def build_agent(self, lr, device, config):
        agent_type = config['type']
        n_agent = config['number']
        algorithm = config['algorithm']
        if agent_type == "trend":
            if 'look_backs' in config.keys():
                look_backs = config['look_backs']
            else:
                min_look_back = config['min_look_back']
                max_look_back = config['max_look_back']
                look_backs = [random.randint(min_look_back, max_look_back) for i in range(n_agent)]
            action_spaces = [(3, 9, 5) for i in range(n_agent)]
            observation_spaces = [look_backs[i]*2 + 2 for i in range(n_agent)]
            config['look_backs'] = look_backs
            agents = [BaseAgent(algorithm = algorithm, observation_space = observation_spaces[i], action_space = action_spaces[i], device = device, look_back = look_backs[i], lr = lr) for i in range(n_agent)]
        elif agent_type == "value":
            pass
        
        return agents

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, config):
        config = deepcopy(config)
        config['Market']['Securities']['TSMC']['price'] = [ round(random.gauss(100, 1), 1) for i in range(100)]
        config['Market']['Securities']['TSMC']['volume'] = [int(random.gauss(100, 10)*10) for i in range(100)]
        config['Market']['Securities']['TSMC']['value'] = [round(random.gauss(100, 1), 1) for i in range(100)]
        self.core = Core(config, market_type="call")
        agent_ids = self.core.multi_env_start(987, 'rl')
        init_states = self.get_states()

        return agent_ids, init_states

    def step(self, actions):
        rewards = []
        next_states = []
        return rewards, next_states

    def get_states(self):
        pass

    def get_state(self, agent_id, look_back):
        market_stats = self.core.get_call_env_state(look_back)
        market = {
            'price': market_stats['price'],
            'volume': market_stats['volume'],
        }

        rl_agent = {'cash': self.core.agent_manager.agents[agent_id].cash,
                    'TSMC': self.core.agent_manager.agents[agent_id].holdings['TSMC'],
                    'wealth': self.core.agent_manager.agents[agent_id].wealth,}
        state = {
            'market': market,
            'agent': rl_agent
        }
        
        return state

    def get_obs(self, agent_id):
        look_back = self.core.agent_manager.agents[agent_id].look_back
        timestep = self.core.timestep

        market_stats = self.core.get_call_env_state(look_back)
        price = {
            'price': market_stats['price'],
            'volume': market_stats['volume'],
        }

        rl_agent = {'cash': self.core.agent_manager.agents[agent_id].cash,
                    'TSMC': self.core.agent_manager.agents[agent_id].holdings['TSMC'],
                    'wealth': self.core.agent_manager.agents[agent_id].wealth,}
        state = {
            'timestep': timestep,
            'price': price,
            'agent': rl_agent
        }
        
        return state

    def close(self):
        orderbooks, agent_manager = self.core.multi_env_close()
        return orderbooks, agent_manager, self.states

    def seed(self, s):
        return s

    def render(self):
        print(f"At: {self.core.timestep}, the market state is:\n{self.core.market.market_stats()}\n")
        if self.core.timestep % 100 == 0:
            print(f"==========={self.core.timestep}===========\n")