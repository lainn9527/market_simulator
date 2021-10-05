import torch
import numpy as np
import math

from algorithm.actor_critic import ActorCritic
from algorithm.ppo_entropy import PPO
from core import agent

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
        self.reward_weight = {'strategy': 0, 'wealth': 1}
        self.timestep = 0

    def get_action(self, state):
        self.timestep += 1
        self.agent_states.append(state['agent'])
        obs = self.obs_wrapper(state)
        action, log_prob = self.rl.get_action(obs)
        action = self.action_wrapper(action, state)
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
        action = self.action_wrapper(action, state)
        return obs, action


    def calculate_reward(self, action, next_state):
        reward = self.reward_wrapper(action, next_state)
        return reward


    def action_wrapper(self, action, state):
        return action

    def obs_wrapper(self, obs):
        return_rate_range = self.look_back
        current_price = obs['market']['price'][-1]
        current_volume = obs['market']['volume'][-1]
        d_p = np.diff(obs['market']['price'][-return_rate_range:]).mean().item()
        return_rate = d_p / current_price

        states = np.array([math.exp(return_rate), current_price, current_volume], np.float32)
        norm_states = (states - states.mean()) / states.std()
        return norm_states

    def reward_dacay(self, decay_rate, strategy_weight, wealth_weight):
        if self.reward_weight['action'] < 0.001:
            return
        decay_weight = decay_rate * self.reward_weight['action']

        self.reward_weight['action'] -= decay_weight
        self.reward_weight['strategy'] += strategy_weight * decay_weight
        self.reward_weight['wealth'] += wealth_weight * decay_weight


    def reward_wrapper(self, action, next_state):
        # strategy reward
        stock_size = 100
        current_price = next_state['market']['price'][-1]
        previous_price = next_state['market']['price'][-self.look_back]
        
        # gap: [-0.5, 0.5] / look_back, stock_ratio [0, 1]
        gap = (current_price - previous_price) / current_price
        stock_ratio = (next_state['agent']['TSMC'] * current_price * stock_size) / next_state['agent']['wealth']
        strategy_reward = 20 * (gap * stock_ratio)


        # volume: 0-5
        # reward: [4 * -0.5 * 5 ,4 * 0.5 * 5]
        bid_or_ask = action[0]
        # tick = action[1] - 4
        # volume = action[2]
        if bid_or_ask == 0:
            strategy_reward = 20 * gap
        elif bid_or_ask == 1:
            strategy_reward = -(20 * gap)
        elif bid_or_ask == 2:
            strategy_reward = -0.02
        
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
        weighted_reward = self.reward_weight['strategy'] * strategy_reward + self.reward_weight['wealth'] * wealth_reward

        # self.reward_dacay(decay_rate = 0.9, strategy_weight = 0.7, wealth_weight = 0.3)
        return {'weighted_reward': wealth_reward, 'strategy_reward': 0, 'wealth_reward': wealth_reward}
        # return strategy_reward

    def final_reward(self):
        pass

    def end_episode(self):
        del self.agent_states[:]
        del self.rl.buffer[:]



class ValueAgent(BaseAgent):
    def __init__(self, algorithm, observation_space, action_space, device, look_back = 1, actor_lr = 1e-3, value_lr = 3e-3, batch_size = 32, buffer_size = 45, n_epoch = 10):
        super().__init__(algorithm, observation_space, action_space, device, look_back, actor_lr=actor_lr, value_lr=value_lr, batch_size=batch_size, buffer_size=buffer_size, n_epoch=n_epoch)
        self.reward_weight = {'strategy': 0, 'wealth': 1}

    def action_wrapper(self, action, state):
        return action
    
    def obs_wrapper(self, obs):
        fundamentalist_discount = 0.75
        current_price = obs['market']['price'][-1]
        current_value = obs['market']['value'][-1]
        fundamentalist_profit = fundamentalist_discount * abs( (current_value - current_price) / current_price)

        utilities = [current_price, current_value, math.exp(fundamentalist_profit)]

        states = np.array(utilities, np.float32)
        norm_states = (states - states.mean()) / states.std()
        return norm_states
 
    def reward_dacay(self, decay_rate, strategy_weight, wealth_weight):
        if self.reward_weight['action'] < 0.001:
            return
        decay_weight = decay_rate * self.reward_weight['action']

        self.reward_weight['action'] -= decay_weight
        self.reward_weight['strategy'] += strategy_weight * decay_weight
        self.reward_weight['wealth'] += wealth_weight * decay_weight

    def reward_wrapper(self, action, next_state):
        # right portfolio to track the value
        # the gap might be around 100, so 150 is pretty large
        stock_size = 100
        present_price = next_state['market']['price'][-1]
        present_value = next_state['market']['value'][-1]
        gap = (present_price - present_value) / present_price
        stock_ratio = (next_state['agent']['TSMC'] * present_price * stock_size) / next_state['agent']['wealth']
        risk_free_rate = next_state['market']['risk_free_rate']

        # stock_ratio [0, 1], gap [-0.5, 0.5]
        # reward 20 * [-0.5, 0.5] = [-10, 10]
        # strategy_reward = 20 * (-gap * stock_ratio)

        # volume: 0-5
        # reward: [4 * -0.5 * 5 ,4 * 0.5 * 5]
        bid_or_ask = action[0]
        # volume = action[2]
        # if bid_or_ask == 0:
        #     strategy_reward = -(4 * gap * volume)
        # elif bid_or_ask == 1:
        #     strategy_reward = 4 * gap * volume
        # if bid_or_ask == 0:
        #     strategy_reward = -(20 * gap)
        # elif bid_or_ask == 1:
        #     strategy_reward = 20 * gap
        # elif bid_or_ask == 2:
        #     strategy_reward = -0.02
        if (gap >= 0 and bid_or_ask == 1) or (gap < 0 and bid_or_ask == 0):
            strategy_reward = 10
        elif bid_or_ask == 2:
            strategy_reward = -0.02
        else:
            strategy_reward = -10

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
        weighted_reward = self.reward_weight['strategy'] * strategy_reward + self.reward_weight['wealth'] * wealth_reward
        # self.reward_dacay(decay_rate = 0.9, strategy_weight = 0.7, wealth_weight = 0.3)

        return {'weighted_reward': wealth_reward, 'strategy_reward': 0, 'wealth_reward': wealth_reward}
        # return strategy_reward

class ScalingAgent(BaseAgent):
    def __init__(self, algorithm, observation_space, action_space, device, look_back=1, actor_lr=0.001, value_lr=0.003, batch_size=32, buffer_size=45, n_epoch=10):
        super().__init__(algorithm, observation_space, action_space, device, look_back=look_back, actor_lr=actor_lr, value_lr=value_lr, batch_size=batch_size, buffer_size=buffer_size, n_epoch=n_epoch)
        self.reward_weight = {'action': 0, 'strategy': 0, 'wealth': 1}
        self.time_delta = 0
        self.v_1 = 2
        self.v_2 = 0.6
        self.beta = 4
        self.alpha_1 = 0.6
        self.alpha_2 = 1.5
        self.alpha_3 = 1
        self.fundamentalist_discount = 0.75
        self.precision = 100
        self.return_rate_range = 20

    def action_wrapper(self, action, state):
        current_price = state['market']['price'][-1]
        current_value = state['market']['value'][-1]

        if action[0] == 0 or action[0] == 1:
            pass
        if action[0] == 2:
            if current_value > current_price:
                action[0] = 0
            elif current_value < current_price:
                action[0] = 1
            else:
                action[0] = 2

        return action


    def obs_wrapper(self, obs):
        # market states with normalization
        dividends = 0
        risk_free_rate = obs['market']['risk_free_rate']
        current_price = obs['market']['price'][-1]
        current_value = obs['market']['value'][-1]
        d_p = np.diff(obs['market']['price'][-self.return_rate_range:]).mean().item()
        return_rate = d_p / current_price

        opt_chartist_profit = (dividends + (1 / self.v_2)*d_p) / current_price - risk_free_rate
        pes_chartist_profit = risk_free_rate - (dividends + (1 / self.v_2)*d_p) / current_price
        fundamentalist_profit = self.fundamentalist_discount * abs( (current_value - current_price) / current_price)

        utility_21 = self.alpha_3 * (opt_chartist_profit - fundamentalist_profit)
        utility_22 = self.alpha_3 * (pes_chartist_profit - fundamentalist_profit)

        utilities = [math.exp(opt_chartist_profit), math.exp(pes_chartist_profit), math.exp(fundamentalist_profit)]

        states = np.array(utilities, np.float32)
        norm_states = (states - states.mean()) / states.std()
        return norm_states
    
    def reward_wrapper(self, action, next_state):
        risk_free_rate = next_state['market']['risk_free_rate']

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
        wealth_reward = self.reward_weight['wealth'] * 10 * (wealth_weight['short']*short_change + wealth_weight['mid']*mid_change + wealth_weight['long']*long_change + wealth_weight['base']*base_change)
        # weighted_reward = self.reward_weight['strategy'] * strategy_reward + self.reward_weight['wealth'] * wealth_reward

        return {'weighted_reward': wealth_reward, 'strategy_reward': 0, 'wealth_reward': wealth_reward}
