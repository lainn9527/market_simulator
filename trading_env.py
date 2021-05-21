import random
import numpy as np
import json
import argparse
import gym
import agent
from time import perf_counter
from collections import defaultdict
from typing import Dict, List
from gym import spaces
from gym.utils import seeding
from datetime import datetime, timedelta
from queue import Queue
from pathlib import Path
from stable_baselines3.ppo import ppo
from stable_baselines3.common.env_checker import check_env
from copy import deepcopy
from core import Core
from market import Market
from order import LimitOrder, MarketOrder
from agent_manager import AgentManager
from utils import write_records
from rl_agent import FeatureExtractor

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
        self.states = []
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
        self.states.append(self.get_obs())
        return self.obs_wrapper(self.states[-1])

    def step(self, action):
        rl_agent_id = self.config['Env']['agent']['id']
        self.core.env_step(action, rl_agent_id = rl_agent_id)
        self.states.append(self.get_obs())
        
        obs = self.obs_wrapper(self.states[-1])
        reward = self.reward_wrapper(rl_agent_id, action)
        done = False
        info = {}

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
        total_reward = 0

        rl_agent = self.core.agent_manager.agents[agent_id]
        if action[0] == 0 or action[0] == 1:
            # VALID_ACTION = 1, INVALID_ACTION = 2, HOLD = 0
            if rl_agent.action_status == 1:
                total_reward += 0.3
            elif rl_agent.action_status == 2:
                total_reward -= 0.2
        elif action[0] == 2:
            total_reward -= 0.02

        mid_steps = 50
        long_steps = 200
        wealth_weight = {'short': 0.15, 'mid': 0.35, 'long': 0.3, 'base': 0.2}
        present_wealth = self.states[-1]['agent']['wealth']
        base_wealth = self.states[0]['agent']['wealth']
        last_wealth = self.states[-2]['agent']['wealth'] if len(self.states) >= 2 else self.states[-1]['agent']['wealth']
        mid_wealth = sum([state['agent']['wealth'] for state in self.states[-1: -(mid_steps+1):-1]]) / len(self.states[-1: -(mid_steps+1):-1])
        long_wealth = sum([state['agent']['wealth'] for state in self.states[-1: -(long_steps+1):-1]]) / len(self.states[-1: -(long_steps+1):-1])

        short_change = (present_wealth - last_wealth) / last_wealth
        mid_change = (present_wealth - mid_wealth) / mid_wealth
        long_change = (present_wealth - long_wealth) / long_wealth
        base_change = (present_wealth - base_wealth) / base_wealth

        total_reward += wealth_weight['short']*short_change + wealth_weight['mid']*mid_change + wealth_weight['long']*long_change + wealth_weight['base']*base_change

        return total_reward

    def get_obs(self):
        obs_config = self.config['Env']['obs']

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
            'orderbook': orderbook,
            'price': price,
            'agent': rl_agent
        }
        
        return state

    def close(self):
        print(f"The wealth of RLAgent is :{self.states[-1]['agent']['wealth']}, gaining {100 * (1 - self.states[-1]['agent']['wealth'] / self.states[0]['agent']['wealth'])}%")
        return self.core.env_close()

    def seed(self):
        return

    def render(self):
        print(f"At: {self.core.timestep}, the market state is:\n{self.core.market.market_stats()}\n")
        if self.core.timestep % 100 == 0:
            print(f"==========={self.core.timestep}===========\n")




if __name__=='__main__':
    config_path = Path("config_zi.json")
    result_dir = Path("rl_result/zi_1000/")
    model_dir = Path("model/")
    if not result_dir.exists():
        result_dir.mkdir(parents=True)
    
    random_seed = 9527
    np.random.seed(random_seed)
    random.seed(random_seed)
    num_of_timesteps = 100
    start_time = perf_counter()

    config = json.loads(config_path.read_text())

    env = TradingEnv(config)
    # check_env(env, warn=True)

    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
    )

    model = ppo.PPO(policy="MultiInputPolicy", env = env, verbose = 1, policy_kwargs = policy_kwargs)
    model.learn(total_timesteps=10000)
    model.save(model_dir / "timestep_10000")
    obs = env.reset()
    n_steps = 1000
    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    env.close()

    # orderbooks, agent_manager = env.close()
    # write_records(orderbooks, agent_manager, result_dir)
    # with open(result_dir / 'config.json', 'w') as fp:
    #     json.dump(config, fp)
    # cost_time = str(timedelta(seconds = perf_counter() - start_time))
    # print(f"Run {num_of_timesteps} in {cost_time}.")    
