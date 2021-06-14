import random
import numpy as np
import gym
from collections import defaultdict
from typing import Dict, List
from gym import spaces
from gym.utils import seeding
from datetime import datetime
from pathlib import Path
from copy import deepcopy

from numpy.lib.arraysetops import setxor1d

from core import agent
from core.core import Core
from core.market import Market
from core.order import LimitOrder, MarketOrder
from core.agent_manager import AgentManager
from core.utils import write_records
from .rl_agent import FeatureExtractor

class TradingEnv(gym.Env):
    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {'render.modes': ['console']}

    def __init__(self, config: Dict):
        super(TradingEnv, self).__init__()
        self.core = None
        self.config = config
        '''
        1. Discrete 3 - BUY[0], SELL[1], HOLD[2]
        2. Discrete 9 - TICK_-4[0], TICK_-3[1], TICK_-2[2], TICK_-1[3], TICK_0[4], TICK_1[5], TICK_2[6], TICK_3[7], TICK_4[8]
        3. Discrete 5 - VOLUME_1[0], VOLUME_2[1], VOLUME_3[2], VOLUME_4[3], VOLUME_5[4],
        '''
        self.action_space = spaces.MultiDiscrete([3, 9, 5])
        # self.observation_space = spaces.Dict({
        #     'orderbook': spaces.Dict({
        #         'bid_volumes': spaces.Box(low = 0, high = 1000, shape=(self.config['obs']['best_price'], )),
        #         'ask_volumes': spaces.Box(low = 0, high = 1000, shape=(self.config['obs']['best_price'], )),
        #     }),
        #     'price': spaces.Dict({
        #         'average': spaces.Box(low = 0, high = 2000, shape=(self.config['obs']['lookback'],)),
        #         'close': spaces.Box(low = 0, high = 2000, shape=(self.config['obs']['lookback'],)),
        #         'best_bid': spaces.Box(low = 0, high = 2000, shape=(self.config['obs']['lookback'],)),
        #         'best_ask': spaces.Box(low = 0, high = 2000, shape=(self.config['obs']['lookback'],)),
        #         'mid': spaces.Box(low = 0, high = 2000, shape=(self.config['obs']['lookback'],)),
        #     }),
        #     'agent': spaces.Box(low = 0, high = 100000000, shape=(3,))
        # })
        self.observation_space = spaces.Dict({
            'orderbook': spaces.Box(low = 0, high = 1000, shape=(2, self.config['Env']['obs']['best_price'], )),
            'price': spaces.Box(low = 0, high = 2000, shape=(5, self.config['Env']['obs']['lookback'],)),
            'agent': spaces.Box(low = 0, high = 100000000, shape=(2,))
        })
        self.start_state = None
        self.states = []
        self.train = True
        # self.reset()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # self.seed(9527)
        config = deepcopy(self.config)
        self.core = Core(config)
        self.core.start_time = datetime.now()
        # register the core for the market and agents
        self.core.random_seed = 9527
        self.core.market.start()
        self.core.agent_manager.start(self.core.market.get_securities())
        self.core.timestep = 0
        self.core.market.open_session()
        for timestep in range(50):
            self.core.step()

        self.core.agent_manager.add_rl_agent(self.config['Env']['agent'])
        print("Set up the following agents:")
        self.start_state = self.get_obs()
        self.states.append(self.start_state)
        return self.obs_wrapper(self.start_state)

    def step(self, action):
        rl_agent_id = self.config['Env']['agent']['id']
        action = action.tolist()
        self.core.env_step(action, rl_agent_id = rl_agent_id)
        self.states.append(self.get_obs())        
        obs = self.obs_wrapper(self.states[-1])
        action_reward, wealth_reward, is_valid = self.reward_wrapper(rl_agent_id, action)
        reward = action_reward + wealth_reward
        done = False
        info = {}

        # record the action & reward in previous state (s_t-1 -> a_t, r_t)
        self.states[-2]['action'] = {'action': action, 'is_valid': is_valid}
        self.states[-2]['reward'] = {'total_reward': reward, 'action_reward': action_reward, 'wealth_reward': wealth_reward}

        # print out the training info
        if self.train and self.core.timestep % 1000 == 0:
            record_states = self.states[-1:-1000:-1]


        return obs, reward, done, info
    
    def obs_wrapper(self, obs):
        price = np.array( [value for value in obs['price'].values()], dtype=np.float32)
        orderbook = np.array( [value for value in obs['orderbook'].values()], dtype=np.float32)
        agent_state = np.array( [value for key, value in obs['agent'].items() if key != 'wealth'], dtype=np.float32)

        # use base price to normalize
        base_price = self.core.market.orderbooks['TSMC'].value
        price = price / base_price
        agent_state[0] = agent_state[0] / 10000

        return {'price': price, 'orderbook': orderbook, 'agent': agent_state}

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
        present_wealth = self.states[-1]['agent']['wealth']
        base_wealth = self.start_state['agent']['wealth']
        last_wealth = self.states[-2]['agent']['wealth'] if len(self.states) >= 2 else self.states[-1]['agent']['wealth']
        mid_wealth = sum([state['agent']['wealth'] for state in self.states[-1: -(mid_steps+1):-1]]) / len(self.states[-1: -(mid_steps+1):-1])
        long_wealth = sum([state['agent']['wealth'] for state in self.states[-1: -(long_steps+1):-1]]) / len(self.states[-1: -(long_steps+1):-1])

        short_change = (present_wealth - last_wealth) / last_wealth
        mid_change = (present_wealth - mid_wealth) / mid_wealth
        long_change = (present_wealth - long_wealth) / long_wealth
        base_change = (present_wealth - base_wealth) / base_wealth

        wealth_reward = wealth_weight['short']*short_change + wealth_weight['mid']*mid_change + wealth_weight['long']*long_change + wealth_weight['base']*base_change

        return action_reward, wealth_reward, is_valid

    def get_obs(self):
        obs_config = self.config['Env']['obs']
        timestep = self.core.timestep

        market_stats = self.core.get_env_state(obs_config['lookback'], obs_config['best_price'])
        bid_volumes = [bid['volume'] for bid in market_stats['bids']] + [0 for i in range(obs_config['best_price'] - len( market_stats['bids']))]
        ask_volumes = [ask['volume'] for ask in market_stats['asks']] + [0 for i in range(obs_config['best_price'] - len( market_stats['asks']))]

        orderbook = {
            'bid_volumes': bid_volumes,
            'ask_volumes': ask_volumes,
        }
        
        # agent_order_ids = self.core.agent_manager.agents['rl'].orders['TSMC']
        # agent_price_volume = defaultdict(int)
        # for order_id in agent_order_ids:
        #     order = self.core.market.orderbooks['TSMC'].orders[order_id].order
        #     agent_price_volume[order['price']] += order['quantity']
        # bid_price = [ price for price in self.orderbooks['TSMC'].bids_price[:5]]
        # ask_price = [ price for price in self.orderbooks['TSMC'].asks_price[:5]]
        price = {
            'average': market_stats['average'],
            'close': market_stats['close'],
            'best_bid': market_stats['best_bid'],
            'best_ask': market_stats['best_ask'],
            'mid': [round( (bid+ask)/2, 2) for bid, ask in zip(market_stats['best_bid'], market_stats['best_ask'],)]
        }

        rl_agent_id = self.config['Env']['agent']['id']
        rl_agent = {'cash': self.core.agent_manager.agents[rl_agent_id].cash,
                    'TSMC': self.core.agent_manager.agents[rl_agent_id].holdings['TSMC'],
                    'wealth': self.core.agent_manager.agents[rl_agent_id].wealth,}
        state = {
            'timestep': timestep,
            'orderbook': orderbook,
            'price': price,
            'agent': rl_agent
        }
        
        return state

    def close(self):
        orderbooks, agent_manager = self.core.env_close()
        return orderbooks, agent_manager, self.states

    def seed(self):
        return

    def render(self):
        print(f"At: {self.core.timestep}, the market state is:\n{self.core.market.market_stats()}\n")
        if self.core.timestep % 100 == 0:
            print(f"==========={self.core.timestep}===========\n")


        