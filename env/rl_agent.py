import torch
import numpy as np
import math
import random

from torch.functional import norm

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
        # ticks = 10
        tick_size = 0.1
        stock_size = 100
        volume_prop = 1
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

        short_change = (present_wealth - short_wealth) / short_wealth
        mid_change = (present_wealth - mid_wealth) / mid_wealth
        long_change = (present_wealth - long_wealth) / long_wealth
        base_change = (present_wealth - base_wealth) / base_wealth

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
        self.reward_weight = {'strategy': 1, 'wealth': 0.4}
        self.timestep = 0


    def obs_wrapper(self, obs):
        return_rate_range = self.look_back
        current_price = obs['market']['price'][-1]
        current_volume = obs['market']['volume'][-1]
        d_p = np.diff(obs['market']['price'][-return_rate_range:]).sum().item()
        return_rate = d_p / current_price

        # agent satate
        cash_prop = obs['agent']['cash'] / obs['agent']['wealth']
        risk_prop = 1-cash_prop
        portfolio = cash_prop - risk_prop # [-1, 1] means [max hold, max cash]

        states = np.array([ portfolio, return_rate * 10])

        return states
        
    def strategy_reward(self, action, next_state):
        pass


    def reward_dacay(self, decay_rate, strategy_weight, wealth_weight):
        decay_weight = decay_rate * self.reward_weight['action']

        self.reward_weight['action'] -= decay_weight
        self.reward_weight['strategy'] += strategy_weight * decay_weight
        self.reward_weight['wealth'] += wealth_weight * decay_weight


    def reward_wrapper(self, action, next_state):
        # action reward
        stock_size = 100
        previous_price = next_state['market']['price'][-2]
        previous_cash = self.agent_states[-1]['cash']
        availabel_ask = self.agent_states[-1]['TSMC']
        available_bid = previous_cash / (stock_size * previous_price)
        if available_bid < 1 and action[0] == 0 or availabel_ask < 1 and action[0] == 1:
            action_reward = -1
        elif action[2] == 2:
            action_reward = -0.01
        else:
            action_reward = 1
        # return {'weighted_reward': action_reward, 'strategy_reward': 0, 'wealth_reward': 0, 'action_reward': action_reward}

        # strategy reward
        risk_free_rate = next_state['market']['risk_free_rate']
        return_rate_range = self.look_back
        current_price = next_state['market']['price'][-1]
        d_p = np.diff(next_state['market']['price'][-return_rate_range-1:-1]).sum().item()
        return_rate = d_p / previous_price
            
        gap = abs(return_rate) - risk_free_rate
        gap = return_rate

        bid_or_ask = action[0]
        ticks = action[1]
        volume = action[2]
        if gap > 0 and bid_or_ask == 0 or gap < 0 and bid_or_ask == 1:
            strategy_reward = abs(gap)
        elif gap > 0 and bid_or_ask == 1 or gap < 0 and bid_or_ask == 0:
            strategy_reward = -abs(gap)
        else:
            strategy_reward = 0

        # holdings
        cash_prop = next_state['agent']['cash'] / next_state['agent']['wealth']
        risk_prop = 1-cash_prop # [-1, 1] means [max hold, max cash]
        portfolio = cash_prop - risk_prop # [-1, 1] means [max hold, max cash]
        if gap > 0:
            # price goes up, bid
            strategy_reward = 10 * abs(gap) * -portfolio
        elif gap < 0:
            # price goes down, ask
            strategy_reward = 10 * abs(gap) * portfolio
        else:
            strategy_reward = 0


        # wealth reward
        wealth_reward = self.get_wealth_reward(next_state)

        reward = {'weighted_reward': wealth_reward, 'strategy_reward': 0, 'wealth_reward': wealth_reward}

        risk_free_rate = next_state['market']['risk_free_rate']     
        present_wealth = next_state['agent']['wealth']
        wealths = [state['wealth'] for state in self.agent_states[-self.look_back:]] + [present_wealth]
        avg_wealth = sum(wealths) / len(wealths)
        first_wealth = wealths[0]
        
        wealth_change = (present_wealth - avg_wealth) / avg_wealth - (pow(1+risk_free_rate, self.look_back) - 1)
        first_wealth_change = (present_wealth - first_wealth) / first_wealth - (pow(1+risk_free_rate, self.look_back) - 1)
        wealth_reward = wealth_change
        wealth_reward = 10 * first_wealth_change
        # weighted_reward = self.reward_weight['strategy'] * strategy_reward + self.reward_weight['wealth'] * wealth_reward
        # reward = {'weighted_reward': weighted_reward, 'strategy_reward': strategy_reward, 'wealth_reward': wealth_reward}
        # reward = {'weighted_reward': strategy_reward, 'strategy_reward': strategy_reward, 'wealth_reward': 0}
        reward = {'weighted_reward': wealth_reward, 'strategy_reward': 0, 'wealth_reward': wealth_reward}
        return reward

    def final_reward(self):
        pass

    def end_episode(self):
        del self.agent_states[:]
        del self.rl.buffer[:]



class ValueAgent(BaseAgent):
    def __init__(self, algorithm, observation_space, action_space, device, look_back = 20, wealth_weight=[0.35, 0.25, 0.25, 0.15], actor_lr = 1e-3, value_lr = 3e-3, batch_size = 32, buffer_size = 45, n_epoch = 10):
        super().__init__(algorithm, observation_space, action_space, device, wealth_weight=wealth_weight, actor_lr=actor_lr, value_lr=value_lr, batch_size=batch_size, buffer_size=buffer_size, n_epoch=n_epoch)
        self.reward_weight = {'strategy': 0.8, 'wealth': 0.2}
        self.fundamentalist_discount = 0.75
        self.look_back = look_back

    def obs_wrapper(self, obs):
        current_price = obs['market']['price'][-1]
        current_value = obs['market']['value'][-1]
        fundamentalist_profit = self.fundamentalist_discount * abs( (current_value - current_price) / current_price)
        gap = (current_value - current_price) / current_price

        # holding
        init_available_bid = self.agent_states[0]['cash'] // (100 * 100)
        init_available_ask = self.agent_states[0]['TSMC']
        stock_size = 100
        cash = obs['agent']['cash']
        available_bid = cash // (current_price * stock_size)
        available_ask = obs['agent']['TSMC']

        cash_prop = obs['agent']['cash'] / obs['agent']['wealth']
        risk_prop = 1-cash_prop
        portfolio = cash_prop - risk_prop # [-1, 1] means [max hold, max cash]

        wealth_utility = self.get_wealth_reward(obs)
        init_price = 100
        init_value = 100

        

        norm_states = np.array( [(available_bid-init_available_bid) / init_available_bid, (available_ask-init_available_ask) / init_available_ask, gap, (current_price - init_price)/init_price, (current_value - init_value) / init_value], np.float32)
        norm_states = np.array( [portfolio, gap, (current_price - init_price)/init_price, (current_value - init_value) / init_value], np.float32)
        norm_states = np.array( [portfolio, gap*10], np.float32)
        # norm_states = np.array( [gap], np.float32)
        # states = np.array( [current_price, current_value, wealth_utility], np.float32)
        # states = np.array( [gap], np.float32)
        # norm_states = (norm_states - norm_states.mean()) / norm_states.std()
        return norm_states
 
    def reward_dacay(self, decay_rate, strategy_weight, wealth_weight):
        if self.reward_weight['action'] < 0.001:
            return
        decay_weight = decay_rate * self.reward_weight['action']

        self.reward_weight['action'] -= decay_weight
        self.reward_weight['strategy'] += strategy_weight * decay_weight
        self.reward_weight['wealth'] += wealth_weight * decay_weight

    def stategy_reward(self, action, next_state):
        present_price = next_state['market']['price'][-2]
        present_value = next_state['market']['value'][-2]
        gap = (present_value - present_price) / present_price

        cash_prop = next_state['agent']['cash'] / next_state['agent']['wealth']
        risk_prop = 1-cash_prop # [-1, 1] means [max hold, max cash]
        portfolio = cash_prop - risk_prop # [-1, 1] means [max hold, max cash]

        if gap > 0:
            # value > price, bid
            # range: [-0.05, 0.05] * [-1, 1]-> [-0.5, 0.5] -> total [-5, 5]
            strategy_reward = abs(gap) * -portfolio
        elif gap < 0:
            # price > value, ask
            strategy_reward = abs(gap) * portfolio
        else:
            strategy_reward = 0

    def reward_wrapper(self, action, next_state):
        present_price = next_state['market']['price'][-2]
        present_value = next_state['market']['value'][-2]
        gap = (present_value - present_price) / present_price
        bid_or_ask = action[0]
        ticks = action[1]
        volume = action[2]


        cash_prop = next_state['agent']['cash'] / next_state['agent']['wealth']
        risk_prop = 1-cash_prop # [-1, 1] means [max hold, max cash]
        portfolio = cash_prop - risk_prop # [-1, 1] means [max hold, max cash]
    

        # wealth reward
        # wealth_reward = self.get_wealth_reward(next_state)

        risk_free_rate = next_state['market']['risk_free_rate']     
        present_wealth = next_state['agent']['wealth']
        wealths = [state['wealth'] for state in self.agent_states[-self.look_back:]] + [present_wealth]
        avg_wealth = sum(wealths) / len(wealths)
        avg_wealth_change = (present_wealth - avg_wealth) / avg_wealth - (pow(1+risk_free_rate, self.look_back) - 1)

        first_wealth = wealths[0]
        first_wealth_change = (present_wealth - first_wealth) / first_wealth - (pow(1+risk_free_rate, self.look_back) - 1)
        wealth_reward = 10 * first_wealth_change

        # weighted_reward = self.reward_weight['strategy'] * strategy_reward + self.reward_weight['wealth'] * wealth_reward
        # reward = {'weighted_reward': wealth_reward, 'strategy_reward': 0, 'wealth_reward': wealth_reward}
        # reward = {'weighted_reward': weighted_reward, 'strategy_reward': strategy_reward, 'wealth_reward': wealth_reward}
        # reward = {'weighted_reward': strategy_reward, 'strategy_reward': strategy_reward, 'wealth_reward': 0}
        reward = {'weighted_reward': wealth_reward, 'strategy_reward': 0, 'wealth_reward': wealth_reward}

        return reward

class ScalingAgent(BaseAgent):
    def __init__(self, algorithm, observation_space, action_space, device, look_back = 20, wealth_weight=[0.35, 0.25, 0.25, 0.15], actor_lr=0.001, value_lr=0.003, batch_size=32, buffer_size=45, n_epoch=10):
        super().__init__(algorithm, observation_space, action_space, device, wealth_weight=wealth_weight, actor_lr=actor_lr, value_lr=value_lr, batch_size=batch_size, buffer_size=buffer_size, n_epoch=n_epoch)
        self.reward_weight = {'action': 0, 'strategy': 0.5, 'wealth': 0.5}
        self.look_back = look_back

        self.fundamentalist_discount = 0.75
        
        self.time_delta = 0
        self.v_1 = 2
        self.v_2 = 0.6
        self.beta = 4
        self.alpha_1 = 0.6
        self.alpha_2 = 1.5
        self.alpha_3 = 1
        self.fundamentalist_discount = 0.75
        self.precision = 100
        
    def obs_wrapper(self, obs):
        # market states with normalization
        eps = 1e-6
        risk_free_rate = obs['market']['risk_free_rate']
        current_price = obs['market']['price'][-1]
        current_value = obs['market']['value'][-1]
        dividends = risk_free_rate * current_value
        d_p = np.diff(obs['market']['price'][-self.look_back:]).sum().item()

        # opt_chartist_profit = (dividends + d_p) / current_price - risk_free_rate
        # pes_chartist_profit = risk_free_rate - (dividends + d_p) / current_price
        # fundamentalist_profit = self.fundamentalist_discount * abs( (current_value - current_price) / current_price)

        # market state
        noise_proft = (dividends + (1 / self.v_2)*d_p) / current_price - risk_free_rate
        fundamentalist_profit = self.fundamentalist_discount * (current_value - current_price) / current_price
        # agent state

        cash_prop = obs['agent']['cash'] / obs['agent']['wealth']
        risk_prop = 1-cash_prop
        portfolio = cash_prop - risk_prop # [-1, 1] means [max hold, max cash]
        norm_states = np.array([portfolio, 10 * noise_proft, 10 * fundamentalist_profit])
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
            # strategy_reward = - (ticks) * abs(gap)
        elif gap > 0 and bid_or_ask == 1 or gap < 0 and bid_or_ask == 0:
            strategy_reward = (ticks + volume) * abs(gap)
            # strategy_reward = (ticks) * abs(gap)
        elif gap == 0 and bid_or_ask != 2:
            strategy_reward = -abs(gap)
        elif bid_or_ask == 2:
            strategy_reward = -0.1
        # strategy_reward *= 10
        # wealth reward
        eps = 1e-6
        risk_free_rate = next_state['market']['risk_free_rate']
        current_price = next_state['market']['price'][-2]
        current_value = next_state['market']['value'][-2]
        dividends = risk_free_rate * current_value
        d_p = np.diff(next_state['market']['price'][-self.look_back-1:-1]).sum().item()

        # opt_chartist_profit = (dividends + d_p) / current_price - risk_free_rate
        # pes_chartist_profit = risk_free_rate - (dividends + d_p) / current_price
        # fundamentalist_profit = self.fundamentalist_discount * abs( (current_value - current_price) / current_price)
        opt_chartist_profit = (dividends + (1 / self.v_2)*d_p) / current_price - risk_free_rate
        pes_chartist_profit = risk_free_rate - (dividends + (1 / self.v_2)*d_p) / current_price
        noise_proft = opt_chartist_profit
        fundamentalist_profit = self.fundamentalist_discount * (current_value - current_price) / current_price
        # holdings

        cash_prop = next_state['agent']['cash'] / next_state['agent']['wealth']
        risk_prop = 1-cash_prop
        portfolio = cash_prop - risk_prop # [-1, 1] means [max hold, max cash]

        if noise_proft >= 0 and fundamentalist_profit >= 0:
            # bid
            strategy_reward = (noise_proft + fundamentalist_profit) * -portfolio
        elif noise_proft >= 0 and fundamentalist_profit <= 0:
            if noise_proft >= abs(fundamentalist_profit):
                # noise, bid
                strategy_reward = (noise_proft + fundamentalist_profit) * -portfolio
            else:
                # fund, ask
                strategy_reward = (fundamentalist_profit + noise_proft) * portfolio
        elif noise_proft < 0 and fundamentalist_profit > 0:
            if abs(noise_proft) >= fundamentalist_profit:
                # noise, ask
                strategy_reward = abs(noise_proft + fundamentalist_profit) * portfolio
            else:
                # fund, bid
                strategy_reward = (fundamentalist_profit + noise_proft) * -portfolio
        elif noise_proft < 0 and fundamentalist_profit < 0:
            strategy_reward = abs(noise_proft + fundamentalist_profit) * portfolio
        strategy_reward = 10 * strategy_reward



        wealth_reward = self.get_wealth_reward(next_state)

        risk_free_rate = next_state['market']['risk_free_rate']     
        present_wealth = next_state['agent']['wealth']
        wealths = [state['wealth'] for state in self.agent_states[-self.look_back:]] + [present_wealth]
        avg_wealth = sum(wealths) / len(wealths)
        first_wealth = wealths[0]
        
        wealth_change = (present_wealth - avg_wealth) / avg_wealth - (pow(1+risk_free_rate, self.look_back) - 1)
        first_wealth_change = (present_wealth - first_wealth) / avg_wealth - (pow(1+risk_free_rate, self.look_back) - 1)
        wealth_reward = wealth_change
        wealth_reward = 10 * first_wealth_change
        # reward = {'weighted_reward': strategy_reward, 'strategy_reward': strategy_reward, 'wealth_reward': 0}
        reward = {'weighted_reward': wealth_reward, 'strategy_reward': 0, 'wealth_reward': wealth_reward}
        # weighted_reward = self.reward_weight['strategy'] * strategy_reward + self.reward_weight['wealth'] * wealth_reward
        # reward = {'weighted_reward': weighted_reward, 'strategy_reward': strategy_reward, 'wealth_reward': wealth_reward}
        return reward
