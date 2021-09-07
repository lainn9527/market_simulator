import torch
import numpy as np
from .algorithm import ActorCritic


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

    def __init__(self, observation_space, action_space, device, look_back):
        self.rl = ActorCritic(observation_space, action_space).to(device)
        self.states = []
        self.device = device
        self.look_back = look_back

    def forward(self, state):

        self.states.append(state)

        obs = self.obs_wrapper(state)
        obs = torch.from_numpy(obs).to(self.device)
        action = self.rl.forward(obs)
        return action

    def calculate_loss(self):
        loss = self.rl.calculate_loss()
        self.rl.clear_memory()
        return loss


    def calculate_reward(self, action, next_state, action_status):
        reward = self.reward_wrapper(action, next_state, action_status)
        total_reward = sum(reward)
        self.rl.rewards.append(total_reward)
        return reward


    def get_parameters(self):
        return self.rl.parameters()

    def obs_wrapper(self, obs):
        price = np.array( [value for value in obs['market']['price']], dtype=np.float32)
        volume = np.array( [value for value in obs['market']['volume']], dtype=np.float32)
        agent_state = np.array( [value for key, value in obs['agent'].items() if key != 'wealth'], dtype=np.float32)
        
        # use base price to normalize
        start_price = self.states[0]['market']['price'][0]
        start_volume = self.states[0]['market']['volume'][0]
        price = price / start_price
        volume = volume / start_volume
        price = price.flatten()
        volume = volume.flatten()
        agent_state[0] = agent_state[0] / (start_price*100)

        # concat
        return np.concatenate([price, volume, agent_state])

    def reward_wrapper(self, action, next_state, action_status):
        # action reward
        is_valid = 1
        action_reward = 0

        if action[0] == 0 or action[0] == 1:
            # VALID_ACTION = 1, INVALID_ACTION = 2, HOLD = 0
            if action_status == 1:
                action_reward += 0.3
            elif action_status == 2:
                action_reward -= 0.2
                is_valid = 0
            
        elif action[0] == 2:
            action_reward -= 0.02

        # wealth reward
        mid_steps = 50
        long_steps = 200
        wealth_weight = {'short': 0.15, 'mid': 0.35, 'long': 0.3, 'base': 0.2}
        present_wealth = next_state['agent']['wealth']
        base_wealth = self.states[0]['agent']['wealth']
        last_wealth = self.states[-1]['agent']['wealth']
        mid_wealths = [present_wealth] + [state['agent']['wealth'] for state in self.states[-1: -(mid_steps+1):-1]]
        mid_wealth = sum(mid_wealths) / len(mid_wealths)
        long_wealths = [present_wealth] + [state['agent']['wealth'] for state in self.states[-1: -(long_steps+1):-1]]
        long_wealth = sum(long_wealths) / len(long_wealths)
        
        short_change = (present_wealth - last_wealth) / last_wealth
        mid_change = (present_wealth - mid_wealth) / mid_wealth
        long_change = (present_wealth - long_wealth) / long_wealth
        base_change = (present_wealth - base_wealth) / base_wealth

        wealth_reward = wealth_weight['short']*short_change + wealth_weight['mid']*mid_change + wealth_weight['long']*long_change + wealth_weight['base']*base_change

        return action_reward, wealth_reward, is_valid

    def final_reward(self):
        pass