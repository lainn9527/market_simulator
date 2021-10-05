import random
import numpy as np
import torch
import math
from copy import deepcopy
from core.core import Core
from .rl_agent import TrendAgent, ValueAgent, ScalingAgent
from core import agent

class MultiTradingEnv:

    def __init__(self):
        self.core = None
        self.agents = []
        self.agent_ids = []
        self.group_agents = {}

    def seed(self, seed=None):
        np.random.seed(seed)
        

    def reset(self, config):
        config = deepcopy(config)
        group_name = list(self.group_agents.keys())
        pre_value = [config['Market']['Securities']['TSMC']['value']]
        pre_price = [config['Market']['Securities']['TSMC']['value']]
        
        for i in range(249):
            pre_value.append(round(math.exp(math.log(pre_value[-1]) + random.gauss(0, 0.005)), 1))
            pre_price.append(pre_price[-1] + round(random.gauss(0, 1), 1))
        
        pre_volume = [int(random.gauss(100, 10)*10) for i in range(249)]
        config['Market']['Securities']['TSMC']['value'] = pre_value
        config['Market']['Securities']['TSMC']['price'] = pre_price
        config['Market']['Securities']['TSMC']['volume'] = pre_volume

        self.core = Core(config, market_type="call")
        agent_ids = self.core.multi_env_start(987, group_name)
        self.agent_ids = agent_ids
        init_states = self.get_states() 

        return agent_ids, init_states

    def step(self, actions):
        done = self.core.multi_env_step(actions)
        next_states = self.get_states()
        next_obs = self.get_obses(next_states)
        rewards = self.get_rewards(actions, next_states)

        return done, rewards, next_states, next_obs

    def close(self):
        orderbooks, agent_manager = self.core.multi_env_close()

        return orderbooks, agent_manager

    def seed(self, s):
        return s

    def render(self, timestep):
        print(f"At: {timestep}, the market state is:\n{self.core.show_market_state()}")

    def build_agents(self,
                     agent_config,
                     actor_lr,
                     value_lr,
                     batch_size,
                     buffer_size,
                     device,
                     resume,
                     resume_model_dir = None):

        device = torch.device(device)
        # build
        
        agents = []
        group_agents = {}
        for config in agent_config:
            name, agent = self.build_agent(actor_lr, value_lr, batch_size, buffer_size, device, config)
            agents += agent
            group_agents[name] = agent
        
        if resume:
            resume_model_path = resume_model_dir / 'model.pkl'
            checkpoint = torch.load(resume_model_path)
            for name, group_agent in group_agents.items():
                for i, agent in enumerate(group_agent):
                    agent.rl.load_state_dict(checkpoint[f"{name}_{i}"])
                print(f"Resume {len(group_agent)} {name} agents from {resume_model_path}.")
        else:
            print(f"Initiate {len(agents)} rl agents")

        self.agents = agents
        self.group_agents = group_agents
        
        return agents

    def build_agent(self, actor_lr, value_lr, batch_size, buffer_size, device, config):
        agent_type = config['type']
        n_agent = config['number']
        algorithm = config['algorithm']
        group_name = f"{config['name']}_{config['number']}"
        agents = []
        min_batch_size = 20
        min_buffer_size = 32

        if agent_type == "trend":
            if 'look_backs' in config.keys():
                look_backs = config['look_backs']
            else:
                min_look_back = config['min_look_back']
                max_look_back = config['max_look_back']
                look_backs = [random.randint(min_look_back, max_look_back) for i in range(n_agent)]
            action_spaces = [(3) for i in range(n_agent)]
            observation_spaces = [3 for i in range(n_agent)]

            # hard code batch size & buffer size
            for i in range(n_agent):
                buffer_size = 45
                batch_size = 32
                n_epoch = 10
                buffer_size = max(min_buffer_size, look_backs[i])
                batch_size = random.randint(min_batch_size, buffer_size)
                n_epoch =  round(14 / (buffer_size / batch_size))
                trend_agent = TrendAgent(algorithm = algorithm,
                                         observation_space = observation_spaces[i],
                                         action_space = action_spaces[i],
                                         device = device,
                                         look_back = look_backs[i], 
                                         actor_lr = actor_lr,
                                         value_lr = value_lr,
                                         batch_size = batch_size,
                                         buffer_size = buffer_size,
                                         n_epoch = n_epoch
                                        )
                agents.append(trend_agent)
            # record
            config['look_backs'] = look_backs
            
        elif agent_type == "value":
            action_spaces = [(3) for i in range(n_agent)]
            observation_spaces = [3 for i in range(n_agent)]
            for i in range(n_agent):
                buffer_size = 45
                batch_size = 32
                n_epoch = 10
                buffer_size = max(min_batch_size, 250)
                batch_size = random.randint(min_batch_size, buffer_size)
                n_epoch =  round(14 / (buffer_size / batch_size))
                value_agent = ValueAgent(algorithm = algorithm,
                                           observation_space = observation_spaces[i],
                                           action_space = action_spaces[i],
                                           device = device,
                                           actor_lr = actor_lr,
                                           value_lr = value_lr,
                                           batch_size = batch_size,
                                           buffer_size = buffer_size,
                                           n_epoch = n_epoch
                                        )

                agents.append(value_agent)

        elif agent_type == "scale":
            action_spaces = [(3) for i in range(n_agent)]
            observation_spaces = [3 for i in range(n_agent)]
            for i in range(n_agent):
                buffer_size = 45
                batch_size = 32
                n_epoch = 10
                buffer_size = max(min_batch_size, 250)
                batch_size = random.randint(min_batch_size, buffer_size)
                n_epoch =  round(14 / (buffer_size / batch_size))
                scale_agent = ScalingAgent(algorithm = algorithm,
                                           observation_space = observation_spaces[i],
                                           action_space = action_spaces[i],
                                           device = device,
                                           actor_lr = actor_lr,
                                           value_lr = value_lr,
                                           batch_size = batch_size,
                                           buffer_size = buffer_size,
                                           n_epoch = n_epoch
                                        )
                agents.append(scale_agent)
        
        return group_name, agents

    def store_agents(self, model_output_path):
        state_dicts = {}
        for name, agents in self.group_agents.items():
            for i, agent in enumerate(agents):
                state_dicts[f"{name}_{i}"] = agent.rl.state_dict()

        torch.save(state_dicts, model_output_path)
        print(f"The model is stored in {model_output_path}")
        
    def get_states(self):
        states = {}
        market_state = self.get_market_state()
        for agent_id, agent in zip(self.agent_ids, self.agents):
            agent_state = self.get_agent_state(agent_id)
            states[agent_id] = {'market': market_state, 'agent': agent_state}

        return states

    def get_market_state(self):
        market_stats = self.core.get_call_env_state(lookback = 99999, from_last = False)
        market = {
            'price': market_stats['price'],
            'volume': market_stats['volume'],
            'value': market_stats['value'],
            'risk_free_rate': market_stats['risk_free_rate'],
            'n_optimistic': agent.ScalingAgent.get_opt_number,
            'n_pessimistic': agent.ScalingAgent.get_pes_number,
            'n_fundamentalist': agent.ScalingAgent.get_fud_number
        }
        return market

    def get_agent_state(self, agent_id):
        agent_state = {'cash': self.core.agent_manager.agents[agent_id].cash,
                    'TSMC': self.core.agent_manager.agents[agent_id].holdings['TSMC'],
                    'wealth': self.core.agent_manager.agents[agent_id].wealth,}
        
        return agent_state

    def get_state(self, agent_id):
        market_stats = self.core.get_call_env_state(lookback = 99999, from_last = False)
        market = {
            'price': market_stats['price'],
            'volume': market_stats['volume'],
            'value': market_stats['value'],
            'risk_free_rate': market_stats['risk_free_rate'],
        }

        rl_agent = {'cash': self.core.agent_manager.agents[agent_id].cash,
                    'TSMC': self.core.agent_manager.agents[agent_id].holdings['TSMC'],
                    'wealth': self.core.agent_manager.agents[agent_id].wealth,}
        state = {
            'market': market,
            'agent': rl_agent
        }
        
        return state
    
    def get_rewards(self, actions, next_states):
        rewards = {}
        for agent, agent_id in zip(self.agents, self.agent_ids):
            rewards[agent_id] = agent.calculate_reward(actions[agent_id], next_states[agent_id])
        
        return rewards

    def get_obses(self, states):
        return {agent_id: agent.obs_wrapper(state) for agent_id, agent, state in zip(self.agent_ids, self.agents, states.values())}
