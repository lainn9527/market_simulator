import random
import numpy as np
import gym
from core import agent
from collections import defaultdict
from typing import Dict, List
from gym import spaces
from gym.utils import seeding
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from core.core import Core




class MultiTradingEnv:

    def __init__(self, config: Dict):
        '''
        Action Space
            1. Discrete 3 - BUY[0], SELL[1], HOLD[2]
            2. Discrete 9 - TICK_-4[0], TICK_-3[1], TICK_-2[2], TICK_-1[3], TICK_0[4], TICK_1[5], TICK_2[6], TICK_3[7], TICK_4[8]
            3. Discrete 5 - VOLUME_1[0], VOLUME_2[1], VOLUME_3[2], VOLUME_4[3], VOLUME_5[4],
        
        State Space
            1. Price and Volume: close, highest, lowest, average price, and volume of previous [lookback] days -> 5 * [lookback]
            2. Agent: cash, holdings -> 2
            Total: 5 * [lookback] + 2
        '''
        self.config = config
        self.train = True
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, look_backs: List):
        # register the core for the market and agents
        config = deepcopy(self.config)
        config['Market']['Securities']['TSMC']['price'] = [random.gauss(100, 1) for i in range(100)]
        config['Market']['Securities']['TSMC']['volume'] = [int(random.gauss(100, 10)*10) for i in range(100)]
        config['Market']['Securities']['TSMC']['value'] = [random.gauss(100, 1) for i in range(100)]
        config['Agent']['RLAgent'][0]['obs']['look_backs'] = look_backs

        self.core = Core(config, market_type="call")
        rl_group_name = f"{config['Agent']['RLAgent'][0]['name']}_{config['Agent']['RLAgent'][0]['number']}"
        self.agents = self.core.multi_env_start(random_seed = 9527, group_name = rl_group_name)

        # make the agent of market simulater the env agent
        self.start_state = {agent_id: self.get_obs(agent_id = agent_id) for agent_id in self.agents}
        self.states = {agent_id: [self.start_state[agent_id]] for agent_id in self.agents}
        agent_obs = {agent_id: self.obs_wrapper(self.start_state[agent_id]) for agent_id in self.agents}

        
        print("Set up the following agents:")
        print(self.agents)

        return self.agents, self.start_state

    def step(self, actions):

        self.core.multi_env_step(actions)
        for agent_id in self.agents:
            self.states[agent_id].append(self.get_obs(agent_id = agent_id))
    
        agents_obs = {agent_id: self.states[agent_id][-1] for agent_id in self.agents}
        
        done = {agent_id: False for agent_id in self.agents}
        info = {}

        # record the action & reward in previous state (s_t-1 -> a_t, r_t)
        for agent_id in self.agents:
            self.states[agent_id][-2]['action'] = {'action': actions[agent_id], 'is_valid': is_valids[agent_id]}
            self.states[agent_id][-2]['reward'] = {'total_reward': rewards[agent_id], 'action_reward': action_rewards[agent_id], 'wealth_reward': wealth_rewards[agent_id]}

        # print out the training info
        # if self.train and self.core.timestep % 1000 == 0:
        #     record_states = self.states[-1:-1000:-1]

        return agents_obs
    

    def get_obs(self, agent_id):
        look_back = self.core.agent_manager.agents[agent_id].look_back
        timestep = self.core.timestep

        market_stats = self.core.get_parallel_env_state(look_back)
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