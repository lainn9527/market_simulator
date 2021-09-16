import random
import numpy as np
import json
import torch

from time import perf_counter
from datetime import timedelta
from typing import Dict
from pathlib import Path
from torch.optim import Adam
from copy import deepcopy
from collections import namedtuple

from core.utils import write_multi_records
from core.core import Core
from env.rl_agent import BaseAgent
from env.multi_env import MultiTradingEnv
Transition = namedtuple('Transition',['state', 'action', 'reward', 'log_prob', 'next_state'])


def train_model(train_config: Dict, env_config: Dict):
    def init_env(config: Dict):
        config = deepcopy(config)
        config['Market']['Securities']['TSMC']['price'] = [ round(random.gauss(100, 1), 1) for i in range(100)]
        config['Market']['Securities']['TSMC']['volume'] = [int(random.gauss(100, 10)*10) for i in range(100)]
        config['Market']['Securities']['TSMC']['value'] = [round(random.gauss(100, 1), 1) for i in range(100)]
        return Core(config, market_type="call")

    def get_state(core, agent_id, look_back):
        market_stats = core.get_call_env_state(look_back)
        market = {
            'price': market_stats['price'],
            'volume': market_stats['volume'],
        }

        rl_agent = {'cash': core.agent_manager.agents[agent_id].cash,
                    'TSMC': core.agent_manager.agents[agent_id].holdings['TSMC'],
                    'wealth': core.agent_manager.agents[agent_id].wealth,}
        state = {
            'market': market,
            'agent': rl_agent
        }
        
        return state

        
    if not train_config['result_dir'].exists():
        train_config['result_dir'].mkdir()
    train_output_dir = train_config['result_dir'] / 'train'

    random_seed = 9527
    np.random.seed(random_seed)
    random.seed(random_seed)

    # init agent parameters
    lr = train_config['lr']
    device = torch.device(train_config['device'])

    # build
    if train_config['resume']:
        resume_config_path = train_config['resume_model_dir'] / 'config.json'
        resume_model_path = train_config['resume_model_dir'] / 'model.pkl'
        resume_config = json.loads(resume_config_path.read_text())
        env_config['Agent']['RLAgent'] = resume_config['Agent']['RLAgent']
        num_rl_agent = env_config['Agent']['RLAgent'][0]['number']
        look_backs = env_config['Agent']['RLAgent'][0]['obs']['look_backs']
        action_spaces = [(3, 9, 5) for i in range(num_rl_agent)]
        observation_spaces = [look_backs[i]*2 + 2 for i in range(num_rl_agent)]
        agents = [BaseAgent(algorithm = 'ppo', observation_space = observation_spaces[i], action_space = action_spaces[i], device = device, look_back = look_backs[i], lr = lr) for i in range(num_rl_agent)]
        checkpoint = torch.load(resume_model_path)
        for i, agent in enumerate(agents):
            agent.rl.load_state_dict(checkpoint[f"base_{i}"])
        print(f"Resume {num_rl_agent} rl agents from {resume_model_path}.")

    else:
        num_rl_agent = env_config['Agent']['RLAgent'][0]['number']
        min_look_back = env_config['Agent']['RLAgent'][0]['obs']['min_look_back']
        max_look_back = env_config['Agent']['RLAgent'][0]['obs']['max_look_back']
        look_backs = [random.randint(min_look_back, max_look_back) for i in range(num_rl_agent)]
        action_spaces = [(3, 9, 5) for i in range(num_rl_agent)]
        observation_spaces = [look_backs[i]*2 + 2 for i in range(num_rl_agent)]
        env_config['Agent']['RLAgent'][0]['obs']['look_backs'] = look_backs
        agents = [BaseAgent(algorithm = 'ppo', observation_space = observation_spaces[i], action_space = action_spaces[i], device = device, look_back = look_backs[i], lr = lr) for i in range(num_rl_agent)]
        print(f"Initiate {num_rl_agent} rl agents.")

    # record the observation spaces of agents
    with open(train_config['result_dir'] / 'config.json', 'w') as fp:
        json.dump(env_config, fp)

        
    # init training parameters
    rl_group_name = f"{env_config['Agent']['RLAgent'][0]['name']}_{env_config['Agent']['RLAgent'][0]['number']}"

    if train_config['train']:
        print("Start training...")
        n_epochs = train_config['train_epochs']
        n_steps = train_config['train_steps']

        for t in range(n_epochs):
            print(f"Epoch {t} start")
            train_env = init_env(env_config)
            agent_ids = train_env.multi_env_start(random_seed, rl_group_name)
            states = {agent_id: get_state(train_env, agent_id, rl_agents[agent_id].look_back) for agent_id in agent_ids}
            rl_records = {agent_id: {'states': [], 'actions': [], 'rewards': []} for agent_id in agent_ids}

            for i in range(n_steps):
                # collect actions
                obs, actions, log_probs, action_status, rewards, next_states = {}, {}, {}, {}, {}, {}
                for agent_id, agent in zip(agent_ids, agents):
                    obs[agent_id], actions[agent_id], log_probs[agent_id] = agent.forward(states[agent_id])

                train_env.multi_env_step(actions)

                for agent_id, agent in zip(agent_ids, agents):
                    next_states[agent_id] = get_state(train_env, agent_id, agent.look_back)
                    next_obs = agent.obs_wrapper(next_states[agent_id])
                    action_status[agent_id] = train_env.get_rl_agent_status(agent_id)
                    rewards[agent_id] = agent.calculate_reward(actions[agent_id], next_states[agent_id], action_status[agent_id])
                    agent.update(Transition(obs[agent_id], actions[agent_id], sum(rewards[agent_id]), log_probs[agent_id], next_obs))

                    # log                
                    rl_records[agent_id]['states'].append(states[agent_id])
                    rl_records[agent_id]['actions'].append(actions[agent_id].tolist())
                    rl_records[agent_id]['rewards'].append(rewards[agent_id])
                
                states = next_states
                if i % 10 == 0:
                    print(f"At: {i}, the market state is:\n{train_env.show_market_state()}")
            
            for agent in agents:
                agent.end_episode()

            orderbooks, agent_manager = train_env.multi_env_close()
            training_record = {'eps': t, 'orderbooks': orderbooks, 'agent_manager': agent_manager, 'states': rl_records}
            write_multi_records(training_record, train_output_dir / f'sim_{t}')    
            print(f"Training result is stored in {train_output_dir / f'sim_{t}'}")

            # store the agents
            model_output_path = train_config['result_dir'] / 'model.pkl'
            state_dicts = {f"base_{i}": agent.rl.state_dict() for i, agent in enumerate(agents)}
            torch.save(state_dicts, model_output_path)
            print(f"The model is stored in {model_output_path}")

    else:
        print("Skip training")

    if train_config['validate']:
        print("Start validating...")
        # validate
        validate_output_dir = train_config['result_dir'] / 'validate'
        n_epochs = train_config['validate_epochs']
        n_steps = train_config['validate_steps']

        for t in range(n_epochs):
            print(f"Epoch {t} start")
            validate_env = init_env(env_config)
            agent_ids = validate_env.multi_env_start(random_seed, rl_group_name)

            # shuffle agent_id
            random.shuffle(agent_ids)
            rl_agents = {agent_id: agent for agent_id, agent in zip(agent_ids, agents)}
            states = {agent_id: get_state(validate_env, agent_id, rl_agents[agent_id].look_back) for agent_id in agent_ids}
            rl_records = {agent_id: {'states': [], 'actions': []} for agent_id in agent_ids}

            for i in range(n_steps):
                actions, action_status, next_states = {}, {}, {}
                actions = {agent_id: agent.predict(states[agent_id]) for agent_id, agent in rl_agents.items()}
                validate_env.multi_env_step(actions)

                for agent_id, agent in rl_agents.items():
                    next_states[agent_id] = get_state(validate_env, agent_id, agent.look_back)
                    next_obs = agent.obs_wrapper(next_states[agent_id])
                    action_status[agent_id] = validate_env.get_rl_agent_status(agent_id)
                    rl_records[agent_id]['states'].append(states[agent_id])
                    rl_records[agent_id]['actions'].append(actions[agent_id].tolist())
                
                states = next_states
                if i % 10 == 0:
                    print(f"At: {i}, the market state is:\n{validate_env.show_market_state()}")
            
            for agent_id, agent in rl_agents.items():
                agent.end_episode()

            orderbooks, agent_manager = validate_env.multi_env_close()
            validate_record = {'eps': t, 'orderbooks': orderbooks, 'agent_manager': agent_manager, 'states': rl_records}
            write_multi_records(validate_record, validate_output_dir / f'sim_{t}')
            print(f"Validation result is stored in {validate_output_dir / f'sim_{t}'}")

    else:
        print("Skip validating")

    print("End the simulation")


if __name__=='__main__':
    model_config = {
        'config_path': Path("config/multi.json"),
        'result_dir': Path("simulation_result/multi/ppo_rl_500/"),
        'resume': True,
        'resume_model_dir': Path("simulation_result/multi/ppo_rl_500/"),
        'train': False,
        'train_epochs': 10,
        'train_steps': 2500,
        'batch_size': 64,
        'lr': 1e-4,
        'device': 'cuda',
        'validate': True,
        'validate_epochs': 3,
        'validate_steps': 500,
    }
    # 4:30 h = 270min = 16200s
    if not model_config['result_dir'].exists():
        model_config['result_dir'].mkdir(parents=True)
    

    start_time = perf_counter()
    env_config = json.loads(model_config['config_path'].read_text())
    # env_config['Agent']['RLAgent'][0]['number'] = 100
    train_model(model_config, env_config)
    

    cost_time = str(timedelta(seconds = perf_counter() - start_time))
    
    
    print(f"Run in {cost_time}.")
