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
from pettingzoo import ParallelEnv
from ray.rllib.env import PettingZooEnv
from core.core import Core
from env.actor_critic import ActorCritic




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
        config['Market']['Securities']['TSMC']['price'] = [random.gauss(100, 10) for i in range(100)]
        config['Market']['Securities']['TSMC']['volume'] = [int(random.gauss(100, 10)*10) for i in range(100)]
        config['Market']['Securities']['TSMC']['value'] = [random.gauss(100, 10) for i in range(100)]

        self.core = Core(config, market_type="call")
        rl_group_name = f"{config['Agent']['RLAgent'][0]['name']}_{config['Agent']['RLAgent'][0]['number']}"
        self.agents = self.core.multi_env_start(random_seed = 9527, group_name = rl_group_name)
        self.look_backs = {agent_id: look_backs[i] for i, agent_id in enumerate(self.agents)}
        observation_spaces = [(2*look_backs[i] + 2) for i in range(len(self.agents))]

        # make the agent of market simulater the env agent
        self.start_state = {agent_id: self.get_obs(agent_id = agent_id) for agent_id in self.agents}
        self.states = {agent_id: [self.start_state[agent_id]] for agent_id in self.agents}
        agent_obs = {agent_id: self.obs_wrapper(self.start_state[agent_id]) for agent_id in self.agents}

        
        print("Set up the following agents:")
        print(self.agents)

        return self.agents, observation_spaces, agent_obs

    def step(self, actions):

        self.core.multi_env_step(actions)
        for agent_id in self.agents:
            self.states[agent_id].append(self.get_obs(agent_id = agent_id))
    
        agents_obs = {agent_id: self.obs_wrapper(self.states[agent_id][-1]) for agent_id in self.agents}
        raw_rewards = {agent_id: self.reward_wrapper(agent_id, actions[agent_id]) for agent_id in self.agents}
        action_rewards = {agent_id: raw_rewards[agent_id][0] for agent_id in self.agents}
        wealth_rewards = {agent_id: raw_rewards[agent_id][1] for agent_id in self.agents}
        is_valids = {agent_id: raw_rewards[agent_id][2] for agent_id in self.agents}
        rewards = {agent_id: (action_rewards[agent_id] + wealth_rewards[agent_id]) for agent_id in self.agents}
        
        done = {agent_id: False for agent_id in self.agents}
        info = {}

        # record the action & reward in previous state (s_t-1 -> a_t, r_t)
        for agent_id in self.agents:
            self.states[agent_id][-2]['action'] = {'action': actions[agent_id], 'is_valid': is_valids[agent_id]}
            self.states[agent_id][-2]['reward'] = {'total_reward': rewards[agent_id], 'action_reward': action_rewards[agent_id], 'wealth_reward': wealth_rewards[agent_id]}

        # print out the training info
        # if self.train and self.core.timestep % 1000 == 0:
        #     record_states = self.states[-1:-1000:-1]

        return agents_obs, rewards
    
    def obs_wrapper(self, obs):
        price = np.array( [value for value in obs['price'].values()], dtype=np.float32)
        agent_state = np.array( [value for key, value in obs['agent'].items() if key != 'wealth'], dtype=np.float32)

        # use base price to normalize
        price[0] = price[0] / self.core.get_base_price('TSMC')
        price[1] = price[1] / self.core.get_base_volume('TSMC')
        price = price.flatten()

        agent_state[0] = agent_state[0] / (self.core.get_base_price('TSMC')*100)
        # concat
        return np.concatenate([price, agent_state])

    def reward_wrapper(self, agent_id, action):
        '''
        Design of reward:
        - action
            - if holdings <= 0 & sell, -0.2
            - if cash < price*100 & buy, -0.2
            - if hold, -0.02
            - if right action, +0.3
            
        - wealth
            - short: present wealth v.s. last step wealth, % * 0.15
            - mid: present wealth v.s. average of last 50 step wealth, % * 0.3
            - long: present wealth v.s. average of last 200 step wealth, % * 0.35
            - origin: present wealth v.s. original wealth, % * 0.2
        '''

        # action reward
        is_valid = 1
        rl_agent = self.core.agent_manager.agents[agent_id]

        action_reward = 0
        if action[0] == 0 or action[0] == 1:
            # VALID_ACTION = 1, INVALID_ACTION = 2, HOLD = 0
            if rl_agent.action_status == 1:
                action_reward += 0.3
            elif rl_agent.action_status == 2:
                action_reward -= 0.2
                is_valid = 0
            
        elif action[0] == 2:
            action_reward -= 0.02

        # wealth reward
        mid_steps = 50
        long_steps = 200
        wealth_weight = {'short': 0.15, 'mid': 0.35, 'long': 0.3, 'base': 0.2}
        present_wealth = self.states[agent_id][-1]['agent']['wealth']
        base_wealth = self.start_state[agent_id]['agent']['wealth']
        last_wealth = self.states[agent_id][-2]['agent']['wealth'] if len(self.states) >= 2 else self.states[agent_id][-1]['agent']['wealth']
        mid_wealth = sum([state['agent']['wealth'] for state in self.states[agent_id][-1: -(mid_steps+1):-1]]) / len(self.states[agent_id][-1: -(mid_steps+1):-1])
        long_wealth = sum([state['agent']['wealth'] for state in self.states[agent_id][-1: -(long_steps+1):-1]]) / len(self.states[agent_id][-1: -(long_steps+1):-1])

        short_change = (present_wealth - last_wealth) / last_wealth
        mid_change = (present_wealth - mid_wealth) / mid_wealth
        long_change = (present_wealth - long_wealth) / long_wealth
        base_change = (present_wealth - base_wealth) / base_wealth

        wealth_reward = wealth_weight['short']*short_change + wealth_weight['mid']*mid_change + wealth_weight['long']*long_change + wealth_weight['base']*base_change

        return action_reward, wealth_reward, is_valid

    def get_obs(self, agent_id):
        look_back = self.look_backs[agent_id]
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