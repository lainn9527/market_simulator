import torch
import numpy as np
import math
import random

from algorithm.actor_critic import ActorCritic
from algorithm.ppo import PPO
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

    def __init__(self, algorithm, observation_space, action_space, device, wealth_weight = [0.35, 0.25, 0.25, 0.15], actor_lr = 1e-3, value_lr = 3e-3, batch_size = 32, buffer_size = 45, n_epoch = 10):
        if algorithm == 'ppo':
            self.rl = PPO(observation_space, action_space, actor_lr, value_lr, batch_size, buffer_size, device, n_epoch).to(device)
        elif algorithm == 'ac':
            self.rl = ActorCritic(observation_space, action_space, actor_lr, value_lr, batch_size, buffer_size, device).to(device)
        self.agent_states = []
        self.wealth_weight = {'short': wealth_weight[0], 'mid': wealth_weight[1], 'long': wealth_weight[2], 'base': wealth_weight[3]}
        self.reward_weight = {'strategy': 0, 'wealth': 0}
        self.timestep = 0

    def get_action(self, state):
        self.timestep += 1
        self.agent_states.append(state['agent'])
        obs = self.obs_wrapper(state)
        action, action_prob = self.rl.get_action(obs)
        # action = self.action_wrapper(action, state)
        action = self.action_wrapper(action, action_prob, state)
        return obs, action, action_prob

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

    def end_episode(self):
        del self.agent_states[:]
        del self.rl.buffer[:]

    # def action_wrapper(self, action, state):
    #     act = action[0]
    #     volume_high = max(1, round(0.1 * state['agent']['TSMC']))
    #     ticks = random.randint(0, 5)
    #     volume = random.randint(1, 5)
    #     return [act, ticks, volume]

    def action_wrapper(self, action, action_prob, state):
        act = action[0]
        current_price = state['market']['price'][-1]
        ticks = round(10 * action_prob)
        tick_size = 0.1
        stock_size = 100
        volume_prop = 0.1
        bid_price = current_price + tick_size * ticks
        available_bid_quantity = state['agent']['cash'] // (bid_price * stock_size)
        available_ask_quantity = state['agent']['TSMC']

        bid_quantity = max(1, round(action_prob * volume_prop * available_bid_quantity))
        ask_quantity = max(1, round(action_prob * volume_prop * available_ask_quantity))

        if act == 0:
            return [act, ticks, bid_quantity]
        elif act == 1:
            return [act, ticks, ask_quantity]
        else:
            return [act, 0, 0]

    # def action_wrapper(self, action, state):
    #     act = action[0]
    #     small_tick_low = 0
    #     small_tick_high = 10
    #     big_tick_low = 10
    #     big_tick_high = 20
    #     small_volume_low = 1
    #     small_volume_high = 10
    #     big_volume_low = 10
    #     big_volume_high = 20

    #     if act == 0:
    #         bid_or_ask = 0
    #         tick = random.randint(small_tick_low, small_tick_high)
    #         volume = random.randint(small_volume_low, small_volume_high)
    #     elif act == 1:
    #         bid_or_ask = 0
    #         tick = random.randint(big_tick_low, big_tick_high)
    #         volume = random.randint(big_volume_low, big_volume_high)
    #     elif act == 2:
    #         bid_or_ask = 1
    #         tick = random.randint(small_tick_low, small_tick_high)
    #         volume = random.randint(small_volume_low, small_volume_high)
    #     elif act == 3:
    #         bid_or_ask = 1
    #         tick = random.randint(big_tick_low, big_tick_high)
    #         volume = random.randint(big_volume_low, big_volume_high)
    #     elif act == 4:
    #         bid_or_ask = 2
    #         tick = 0
    #         volume = 0

    #     return [bid_or_ask, tick, volume]

        
    def obs_wrapper(self, obs):
        pass

    def reward_dacay(self, decay_rate, strategy_weight, wealth_weight):
        if self.reward_weight['action'] < 0.001:
            return
        decay_weight = decay_rate * self.reward_weight['action']

        self.reward_weight['action'] -= decay_weight
        self.reward_weight['strategy'] += strategy_weight * decay_weight
        self.reward_weight['wealth'] += wealth_weight * decay_weight

    def get_wealth_reward(self, next_state):
        risk_free_rate = next_state['market']['risk_free_rate']
        wealth_weight = self.wealth_weight
        short_steps = min(20, self.timestep)
        mid_steps = min(60, self.timestep)
        long_steps = min(250, self.timestep)
        total_steps = self.timestep
        present_wealth = next_state['agent']['wealth']
        base_wealth = self.agent_states[0]['wealth']
        short_wealths = [present_wealth] + [state['wealth'] for state in self.agent_states[-short_steps:]]
        short_wealth = sum(short_wealths) / len(short_wealths)
        mid_wealths = [present_wealth] + [state['wealth'] for state in self.agent_states[-mid_steps:]]
        mid_wealth = sum(mid_wealths) / len(mid_wealths)
        long_wealths = [present_wealth] + [state['wealth'] for state in self.agent_states[-long_steps:]]
        long_wealth = sum(long_wealths) / len(long_wealths)
        
        short_change = (present_wealth - short_wealth) / short_wealth - (pow(1+risk_free_rate, short_steps) - 1)
        mid_change = (present_wealth - mid_wealth) / mid_wealth - (pow(1+risk_free_rate, mid_steps) - 1)
        long_change = (present_wealth - long_wealth) / long_wealth - (pow(1+risk_free_rate, long_steps) - 1)
        base_change = (present_wealth - base_wealth) / base_wealth - (pow(1+risk_free_rate, total_steps) - 1)

        # short_change = (present_wealth - short_wealths) / short_wealths - short_steps * risk_free_rate
        # mid_change = (present_wealth - mid_wealth) / mid_wealth - mid_steps * risk_free_rate
        # long_change = (present_wealth - long_wealth) / long_wealth - long_steps * risk_free_rate
        # base_change = (present_wealth - base_wealth) / base_wealth - total_steps * risk_free_rate

        wealth_reward = wealth_weight['short']*short_change + wealth_weight['mid']*mid_change + wealth_weight['long']*long_change + wealth_weight['base']*base_change
        return wealth_reward


    def reward_wrapper(self, action, next_state):
        pass

    def final_reward(self):
        pass


class TrendAgent(BaseAgent):
    def __init__(self, algorithm, observation_space, action_space, device, look_back=1, wealth_weight=[0.35, 0.25, 0.25, 0.15], actor_lr=0.001, value_lr=0.003, batch_size=32, buffer_size=45, n_epoch=10):
        super().__init__(algorithm, observation_space, action_space, device, wealth_weight=wealth_weight, actor_lr=actor_lr, value_lr=value_lr, batch_size=batch_size, buffer_size=buffer_size, n_epoch=n_epoch)
        self.look_back = look_back
        self.reward_weight = {'strategy': 0.6, 'wealth': 0.4}
        self.timestep = 0


    def obs_wrapper(self, obs):
        return_rate_range = self.look_back
        current_price = obs['market']['price'][-1]
        current_volume = obs['market']['volume'][-1]
        d_p = np.diff(obs['market']['price'][-return_rate_range:]).sum().item()
        return_rate = d_p / current_price

        wealth_utility = self.get_wealth_reward(obs)
        states = np.array([math.exp(return_rate), current_price, current_volume, wealth_utility], np.float32)
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
        risk_free_rate = next_state['market']['risk_free_rate']
        return_rate_range = self.look_back
        current_price = next_state['market']['price'][-1]
        d_p = np.diff(next_state['market']['price'][-return_rate_range:]).mean().item()
        return_rate = d_p / current_price
            
        gap = abs(return_rate) - risk_free_rate

        bid_or_ask = action[0]
        ticks = action[1]
        volume = action[2]
        if gap > 0 and bid_or_ask == 0 or gap < 0 and bid_or_ask == 1:
            strategy_reward = (ticks + volume) * abs(gap)
        elif gap > 0 and bid_or_ask == 1 or gap < 0 and bid_or_ask == 0:
            strategy_reward = -(ticks + volume) * abs(gap)
        elif gap == 0 and bid_or_ask != 2:
            strategy_reward = -abs(gap)
        elif bid_or_ask == 2:
            if gap == 0:
                strategy_reward = abs(gap)
            else:
                strategy_reward = -abs(gap) * 5


        # wealth reward
        wealth_reward = self.get_wealth_reward(next_state)

        reward = {'weighted_reward': wealth_reward, 'strategy_reward': 0, 'wealth_reward': wealth_reward}
        
        weighted_reward = self.reward_weight['strategy'] * strategy_reward + self.reward_weight['wealth'] * wealth_reward
        reward = {'weighted_reward': weighted_reward, 'strategy_reward': strategy_reward, 'wealth_reward': wealth_reward}
        return reward

    def final_reward(self):
        pass

    def end_episode(self):
        del self.agent_states[:]
        del self.rl.buffer[:]



class ValueAgent(BaseAgent):
    def __init__(self, algorithm, observation_space, action_space, device, wealth_weight=[0.35, 0.25, 0.25, 0.15], actor_lr = 1e-3, value_lr = 3e-3, batch_size = 32, buffer_size = 45, n_epoch = 10):
        super().__init__(algorithm, observation_space, action_space, device, wealth_weight=wealth_weight, actor_lr=actor_lr, value_lr=value_lr, batch_size=batch_size, buffer_size=buffer_size, n_epoch=n_epoch)
        self.reward_weight = {'strategy': 0.8, 'wealth': 0.2}
        self.fundamentalist_discount = 0.75

    def obs_wrapper(self, obs):
        current_price = obs['market']['price'][-1]
        current_value = obs['market']['value'][-1]
        fundamentalist_profit = self.fundamentalist_discount * abs( (current_value - current_price) / current_price)
        gap = (current_price - current_value) / current_price

        wealth_utility = self.get_wealth_reward(obs)
        states = np.array( [current_price, current_value, math.exp(fundamentalist_profit), wealth_utility], np.float32)
        # states = np.array( [current_price, current_value, wealth_utility], np.float32)
        # states = np.array( [gap], np.float32)
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
        present_price = next_state['market']['price'][-1]
        present_value = next_state['market']['value'][-1]
        gap = (present_price - present_value) / present_price
        bid_or_ask = action[0]
        ticks = action[1]
        volume = action[2]

        if gap > 0 and bid_or_ask == 0 or gap < 0 and bid_or_ask == 1:
            strategy_reward = - abs(gap)
        elif gap > 0 and bid_or_ask == 1 or gap < 0 and bid_or_ask == 0:
            strategy_reward = abs(gap)
        else:
            strategy_reward = 0



        # wealth reward
        wealth_reward = self.get_wealth_reward(next_state)

        weighted_reward = self.reward_weight['strategy'] * strategy_reward + self.reward_weight['wealth'] * wealth_reward
        reward = {'weighted_reward': weighted_reward, 'strategy_reward': strategy_reward, 'wealth_reward': wealth_reward}
        # reward = {'weighted_reward': wealth_reward, 'strategy_reward': 0, 'wealth_reward': wealth_reward}
        # reward = {'weighted_reward': strategy_reward, 'strategy_reward': strategy_reward, 'wealth_reward': 0}
        return reward

class ScalingAgent(BaseAgent):
    def __init__(self, algorithm, observation_space, action_space, device, wealth_weight=[0.35, 0.25, 0.25, 0.15], actor_lr=0.001, value_lr=0.003, batch_size=32, buffer_size=45, n_epoch=10):
        super().__init__(algorithm, observation_space, action_space, device, wealth_weight=wealth_weight, actor_lr=actor_lr, value_lr=value_lr, batch_size=batch_size, buffer_size=buffer_size, n_epoch=n_epoch)
        self.reward_weight = {'action': 0, 'strategy': 0.6, 'wealth': 0.4}
        self.time_delta = 0
        self.fundamentalist_discount = 0.75
        self.return_rate_range = 20
        
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

    def action_wrapper(self, action, action_prob, state):
        act = action[0]
        current_price = state['market']['price'][-1]
        current_value = state['market']['value'][-1]
        ticks = round(10 * action_prob)
        tick_size = 0.1
        stock_size = 100
        volume_prop = 0.1
        bid_price = current_price + tick_size * ticks
        available_bid_quantity = state['agent']['cash'] // (bid_price * stock_size)
        available_ask_quantity = state['agent']['TSMC']

        bid_quantity = max(1, round(action_prob * volume_prop * available_bid_quantity))
        ask_quantity = max(1, round(action_prob * volume_prop * available_ask_quantity))

        if act == 0:
            return [act, ticks, bid_quantity]
        elif act == 1:
            return [act, ticks, ask_quantity]

        elif act == 2:
            if current_price > current_value:
                return [1, ticks, ask_quantity]
            elif current_price < current_value:
                return [0, ticks, bid_quantity]
            else:                
                return [2, 0, 0]
    
    def obs_wrapper(self, obs):
        # market states with normalization
        eps = 1e-6
        risk_free_rate = obs['market']['risk_free_rate']
        current_price = obs['market']['price'][-1]
        current_value = obs['market']['value'][-1]
        dividends = risk_free_rate * current_value
        d_p = np.diff(obs['market']['price'][-self.return_rate_range:]).sum().item()

        # opt_chartist_profit = (dividends + d_p) / current_price - risk_free_rate
        # pes_chartist_profit = risk_free_rate - (dividends + d_p) / current_price
        # fundamentalist_profit = self.fundamentalist_discount * abs( (current_value - current_price) / current_price)
        opt_chartist_profit = (dividends + (1 / self.v_2)*d_p) / current_price - risk_free_rate
        pes_chartist_profit = risk_free_rate - (dividends + (1 / self.v_2)*d_p) / current_price
        fundamentalist_profit = self.fundamentalist_discount * (current_value - current_price) / current_price

        wealth_utility = self.get_wealth_reward(obs)
        utilities = [math.exp(opt_chartist_profit), math.exp(pes_chartist_profit), math.exp(fundamentalist_profit)]
        # utilities = [opt_chartist_profit, pes_chartist_profit, fundamentalist_profit]
        states = np.array(utilities, np.float32)
        print(states)
        norm_states = (states - states.mean()) / (states.std() + eps)
        return norm_states
    
    def reward_wrapper(self, action, next_state):
        present_price = next_state['market']['price'][-2]
        present_value = next_state['market']['value'][-2]
        gap = (present_price - present_value) / present_price
        bid_or_ask = action[0]
        ticks = action[1]
        volume = action[2]

        if gap > 0 and bid_or_ask == 0 or gap < 0 and bid_or_ask == 1:
            strategy_reward = - (ticks + volume) * abs(gap)
            strategy_reward = - (ticks) * abs(gap)
        elif gap > 0 and bid_or_ask == 1 or gap < 0 and bid_or_ask == 0:
            strategy_reward = (ticks + volume) * abs(gap)
            strategy_reward = (ticks) * abs(gap)
        elif gap == 0 and bid_or_ask != 2:
            strategy_reward = -abs(gap)
        elif bid_or_ask == 2:
            strategy_reward = -0.1
        strategy_reward *= 10
        # wealth reward
        wealth_reward = self.get_wealth_reward(next_state)

        # reward = {'weighted_reward': strategy_reward, 'strategy_reward': strategy_reward, 'wealth_reward': 0}
        reward = {'weighted_reward': wealth_reward, 'strategy_reward': 0, 'wealth_reward': wealth_reward}
        # weighted_reward = self.reward_weight['strategy'] * strategy_reward + self.reward_weight['wealth'] * wealth_reward
        # reward = {'weighted_reward': weighted_reward, 'strategy_reward': strategy_reward, 'wealth_reward': wealth_reward}
        return reward