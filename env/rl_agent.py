import torch
import numpy as np

from algorithm.actor_critic import ActorCritic
from algorithm.ppo import PPO

class BaseAgent:
    '''
    Design of agent

    Observation Sapce
    - market
        - price
        - volume

    - agent state
        - cash
        - holdings
        - wealth


    Action Space
    - Discrete 3 - BUY[0], SELL[1], HOLD[2]
    - Discrete 9 - TICK_-4[0], TICK_-3[1], TICK_-2[2], TICK_-1[3], TICK_0[4], TICK_1[5], TICK_2[6], TICK_3[7], TICK_4[8]
    - Discrete 5 - VOLUME_1[0], VOLUME_2[1], VOLUME_3[2], VOLUME_4[3], VOLUME_5[4],

    Reward:
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

    def __init__(self, algorithm, observation_space, action_space, device, look_back = 1, actor_lr = 1e-3, value_lr = 3e-3, batch_size = 32, buffer_size = 45, n_epoch = 10):
        if algorithm == 'ppo':
            self.rl = PPO(observation_space, action_space, actor_lr, value_lr, batch_size, buffer_size, device, n_epoch).to(device)
        elif algorithm == 'ac':
            self.rl = ActorCritic(observation_space, action_space, actor_lr, value_lr, batch_size, buffer_size, device).to(device)
        self.agent_states = []
        self.look_back = look_back
        self.reward_weight = {'action': 0.5, 'strategy': 0, 'wealth': 0.5}
        self.timestep = 0

    def get_action(self, state):
        self.timestep += 1
        self.agent_states.append(state['agent'])
        obs = self.obs_wrapper(state)
        action, log_prob = self.rl.get_action(obs)
        action = self.action_wrapper(action)
        return obs, action, log_prob

    def update(self, transition):
        self.rl.buffer.append(transition)
        if len(self.rl.buffer) % self.rl.buffer_size == 0 and len(self.rl.buffer) > 0:
            return self.rl.update()
        else:
            return None


    def predict(self, state):
        self.agent_states.append(state['agent'])
        obs = self.obs_wrapper(state)
        action = self.rl.predict(obs)
        action = self.action_wrapper(action)
        return action


    def calculate_reward(self, action, next_state, action_status):
        reward = self.reward_wrapper(action, next_state, action_status)
        return reward


    def action_wrapper(self, action):
        return action

    def obs_wrapper(self, obs):
        # market
        price = np.array( [value for value in obs['market']['price'][-self.look_back:]], dtype=np.float32)
        volume = np.array( [value for value in obs['market']['volume'][-self.look_back:]], dtype=np.float32)
        
        current_price = price[-1]
        ma = price.mean()
        gap = (current_price - ma) / current_price / len(price)
        mean_volume = volume.mean().item()
        

        # base_price = obs['market']['price'][0]
        # price = price / base_price
        # base_volume = obs['market']['volume'][0]
        # volume = volume / base_volume
        # price = price.flatten()
        # volume = volume.flatten()

        # agent
        cash = obs['agent']['cash'] / self.agent_states[0]['cash']
        holdings = obs['agent']['TSMC'] / self.agent_states[0]['TSMC']
        wealth = obs['agent']['wealth'] / self.agent_states[0]['wealth']
        agent_state = np.array([cash, holdings, wealth], np.float32)

        # concat
        return np.concatenate([price, volume, agent_state])
        # return {'pv': np.stack([price, volume]), 'agent_state': agent_state}

    def reward_dacay(self, decay_rate, strategy_weight, wealth_weight):
        if self.reward_weight['action'] < 0.001:
            return
        decay_weight = decay_rate * self.reward_weight['action']

        self.reward_weight['action'] -= decay_weight
        self.reward_weight['strategy'] += strategy_weight * decay_weight
        self.reward_weight['wealth'] += wealth_weight * decay_weight


    def reward_wrapper(self, action, next_state, action_status):
        # action reward
        action_reward = 0
        
        if action[0] == 0 or action[0] == 1:
            # VALID_ACTION = 1, INVALID_ACTION = 2, HOLD = 0
            if action_status == 1:
                action_reward += 6
            elif action_status == 2:
                action_reward -= 4
            
        elif action[0] == 2:
            action_reward -= 0.4

        action_reward = self.reward_weight['action'] * action_reward
        
        # strategy reward



        # wealth reward
        risk_free_rate = next_state['market']['risk_free_rate']
        short_steps = 20
        mid_steps = 60
        long_steps = 250
        total_steps = len(next_state['market']['price'])
        wealth_weight = {'short': 0.35, 'mid': 0.25, 'long': 0.25, 'base': 0.15}
        present_wealth = next_state['agent']['wealth']
        base_wealth = self.agent_states[0]['wealth']
        short_wealths = [present_wealth] + [state['wealth'] for state in self.agent_states[-short_steps:]]
        short_wealths = sum(short_wealths) / len(short_wealths)
        mid_wealths = [present_wealth] + [state['wealth'] for state in self.agent_states[-mid_steps:]]
        mid_wealth = sum(mid_wealths) / len(mid_wealths)
        long_wealths = [present_wealth] + [state['wealth'] for state in self.agent_states[-long_steps:]]
        long_wealth = sum(long_wealths) / len(long_wealths)
        
        short_change = (present_wealth - short_wealths) / short_wealths - short_steps * risk_free_rate
        mid_change = (present_wealth - mid_wealth) / mid_wealth - mid_steps * risk_free_rate
        long_change = (present_wealth - long_wealth) / long_wealth - long_steps * risk_free_rate
        base_change = (present_wealth - base_wealth) / base_wealth - total_steps * risk_free_rate

        # 2 times wealth got reward 10
        wealth_reward = self.reward_weight['wealth'] * 10 * (wealth_weight['short']*short_change + wealth_weight['mid']*mid_change + wealth_weight['long']*long_change + wealth_weight['base']*base_change)
        weighted_reward = action_reward + wealth_reward

        self.reward_dacay(decay_rate = 0.9, strategy_weight = 0, wealth_weight = 1)
        return {'weighted_reward': weighted_reward, 'action_reward': action_reward, 'wealth_reward': wealth_reward}

    def final_reward(self):
        pass

    def end_episode(self):
        del self.agent_states[:]
        del self.rl.buffer[:]



class ValueAgent(BaseAgent):
    def __init__(self, algorithm, observation_space, action_space, device, look_back = 1, actor_lr = 1e-3, value_lr = 3e-3, batch_size = 32, buffer_size = 45, n_epoch = 10):
        super().__init__(algorithm, observation_space, action_space, device, look_back, actor_lr=actor_lr, value_lr=value_lr, batch_size=batch_size, buffer_size=buffer_size, n_epoch=n_epoch)
        self.reward_weight = {'action': 0.4, 'strategy': 0.3, 'wealth': 0.3}

    def action_wrapper(self, action):
        return action
    
    def obs_wrapper(self, obs):
        # market states with normalization
        price = obs['market']['price'][-1] / obs['market']['price'][0]
        volume = obs['market']['volume'][-1] / obs['market']['volume'][0]
        value = obs['market']['value'][-1] / obs['market']['value'][0]
        risk_free_rate = obs['market']['risk_free_rate']
        market_state = np.array([price, volume, value, risk_free_rate], dtype = np.float32)
        
        # agent states
        cash = obs['agent']['cash'] / self.agent_states[0]['cash']
        holdings = obs['agent']['TSMC'] / self.agent_states[0]['TSMC']
        wealth = obs['agent']['wealth'] / self.agent_states[0]['wealth']
        agent_state = np.array([cash, holdings, wealth], np.float32)
        return np.concatenate([market_state, agent_state])
 
    def reward_dacay(self, decay_rate, strategy_weight, wealth_weight):
        if self.reward_weight['action'] < 0.001:
            return
        decay_weight = decay_rate * self.reward_weight['action']

        self.reward_weight['action'] -= decay_weight
        self.reward_weight['strategy'] += strategy_weight * decay_weight
        self.reward_weight['wealth'] += wealth_weight * decay_weight

    def reward_wrapper(self, action, next_state, action_status):
        # proper action reward
        action_reward = 0
        if action[0] == 0 or action[0] == 1:
            # VALID_ACTION = 1, INVALID_ACTION = 2, HOLD = 0
            if action_status == 1:
                action_reward += 6
            elif action_status == 2:
                action_reward -= 4
            
        elif action[0] == 2:
            action_reward -= 0.4
        
        action_reward = self.reward_weight['action'] * action_reward
        # right portfolio to track the value
        # the gap might be around 100, so 150 is pretty large
        stock_size = 100
        present_price = next_state['market']['price'][-1]
        present_value = next_state['market']['value'][-1]
        gap = (present_price - present_value) / present_price
        stock_ratio = (next_state['agent']['TSMC'] * present_price * stock_size) / next_state['agent']['wealth']
        risk_free_rate = next_state['market']['risk_free_rate']

        # holdings [0, 1], gap [-0.5, 0.5]
        # reward 20 * [-0.5, 0.5] = [-10, 10]
        strategy_reward = 20 * (-gap * stock_ratio)

        # wealth reward
        short_steps = 20
        mid_steps = 60
        long_steps = 250
        total_steps = len(next_state['market']['price'])
        wealth_weight = {'short': 0.35, 'mid': 0.25, 'long': 0.25, 'base': 0.15}
        present_wealth = next_state['agent']['wealth']
        base_wealth = self.agent_states[0]['wealth']
        short_wealths = [present_wealth] + [state['wealth'] for state in self.agent_states[-short_steps:]]
        short_wealths = sum(short_wealths) / len(short_wealths)
        mid_wealths = [present_wealth] + [state['wealth'] for state in self.agent_states[-mid_steps:]]
        mid_wealth = sum(mid_wealths) / len(mid_wealths)
        long_wealths = [present_wealth] + [state['wealth'] for state in self.agent_states[-long_steps:]]
        long_wealth = sum(long_wealths) / len(long_wealths)
        
        short_change = (present_wealth - short_wealths) / short_wealths - short_steps * risk_free_rate
        mid_change = (present_wealth - mid_wealth) / mid_wealth - mid_steps * risk_free_rate
        long_change = (present_wealth - long_wealth) / long_wealth - long_steps * risk_free_rate
        base_change = (present_wealth - base_wealth) / base_wealth - total_steps * risk_free_rate

        # 2 times wealth got reward 10, so the range might be [-10, 10]

        wealth_reward = 10 * (wealth_weight['short']*short_change + wealth_weight['mid']*mid_change + wealth_weight['long']*long_change + wealth_weight['base']*base_change)

        # weight sum
        weighted_reward = self.reward_weight['action'] * action_reward + self.reward_weight['strategy'] * strategy_reward + self.reward_weight['wealth'] * wealth_reward
        self.reward_dacay(decay_rate = 0.9, strategy_weight = 0.7, wealth_weight = 0.3)

        return {'weighted_reward': weighted_reward, 'action_reward': action_reward, 'strategy_reward': strategy_reward, 'wealth_reward': wealth_reward}
 

 