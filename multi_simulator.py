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

Transition = namedtuple('Transition',['state', 'action', 'reward', 'log_prob', 'next_state', 'done'])
def train_model(train_config: Dict, env_config: Dict):
    if not train_config['result_dir'].exists():
        train_config['result_dir'].mkdir()
    train_output_dir = train_config['result_dir'] / 'train'

    random_seed = 9527
    np.random.seed(random_seed)
    random.seed(random_seed)

    # check if resume
    if train_config['resume']:
        resume_config_path = train_config['resume_model_dir'] / 'config.json'
        env_config = json.loads(resume_config_path.read_text())

    # init training env
    multi_env = MultiTradingEnv()
    rl_agent_config = env_config['Agent']['RLAgent']
    agents = multi_env.build_agents(rl_agent_config,
                                    train_config['actor_lr'],
                                    train_config['value_lr'],
                                    train_config['batch_size'],
                                    train_config['buffer_size'],
                                    train_config['device'],
                                    train_config['resume'],
                                    train_config['resume_model_dir']
                                )
        

    # record the observation spaces of agents
    with open(train_config['result_dir'] / 'config.json', 'w') as fp:
        json.dump(env_config, fp)
        
    # init training parameters
    if train_config['train']:
        print("Start training...")
        n_epochs = train_config['train_epochs']
        n_steps = train_config['train_steps']

        for t in range(n_epochs):
            print(f"Epoch {t} start")
            agent_ids, states = multi_env.reset(env_config)
            rl_records = {agent_id: {'states': [], 'actions': [], 'rewards': [], 'policy_loss': [], 'value_loss': []} for agent_id in agent_ids}

            for i in range(n_steps):
                # collect actions
                obs, actions, log_probs, rewards = {}, {}, {}, {}
                for agent_id, agent in zip(agent_ids, agents):
                    obs[agent_id], actions[agent_id], log_probs[agent_id] = agent.get_action(states[agent_id])

                # rewards, next_steps
                done, rewards, next_states, next_obs = multi_env.step(actions)

                for agent_id, agent in zip(agent_ids, agents):
                    loss = agent.update(Transition(obs[agent_id],actions[agent_id], rewards[agent_id]['weighted_reward'], log_probs[agent_id], next_obs[agent_id], done))

                    # log
                    rl_records[agent_id]['states'].append(states[agent_id]['agent'])
                    rl_records[agent_id]['actions'].append(actions[agent_id])
                    rl_records[agent_id]['rewards'].append(rewards[agent_id])
                    
                    if loss != None:
                        rl_records[agent_id]['policy_loss'] += loss['policy_loss']
                        rl_records[agent_id]['value_loss'] += loss['value_loss']
                
                states = next_states
                if i % 10 == 0:
                    multi_env.render(i)
                
                if done:
                    multi_env.render(i)
                    print(f"No quote. End of this episode")
                    break
            
            for agent_id, agent in zip(agent_ids, agents):
                agent.end_episode()

            # log
            orderbooks, agent_manager = multi_env.close()
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
            agent_ids, states = multi_env.reset(env_config)
            rl_records = {agent_id: {'states': [], 'actions': [], 'rewards': []} for agent_id in agent_ids}

            for i in range(n_steps):
                # collect actions
                actions = {}
                for agent_id, agent in zip(agent_ids, agents):
                    actions[agent_id] = agent.predict(states[agent_id])

                # rewards, next_steps
                done, rewards, next_states, next_obs = multi_env.step(actions)

                for agent_id, agent in zip(agent_ids, agents):
                    # log                
                    rl_records[agent_id]['states'].append(states[agent_id]['agent'])
                    rl_records[agent_id]['actions'].append(actions[agent_id])
                
                states = next_states
                if i % 10 == 0:
                    multi_env.render(i)

                if done:
                    multi_env.render(i)
                    print(f"No quote. End of this episode")
                    break    
            
            for agent_id, agent in zip(agent_ids, agents):
                agent.end_episode()

            orderbooks, agent_manager = multi_env.close()
            validate_record = {'eps': t, 'orderbooks': orderbooks, 'agent_manager': agent_manager, 'states': rl_records}
            write_multi_records(validate_record, validate_output_dir / f'sim_{t}')    
            print(f"Validation result is stored in {validate_output_dir / f'sim_{t}'}")

    else:
        print("Skip validating")

    print("End the simulation")


if __name__=='__main__':
    model_config = {
        'config_path': Path("config/all_100.json"),
        'result_dir': Path("simulation_result/multi/all_100/"),
        'resume': False,
        'resume_model_dir': Path("simulation_result/multi/price_500/"),
        'train': True,
        'train_epochs': 10,
        'train_steps': 2500,
        'validate': True,
        'validate_epochs': 10,
        'validate_steps': 2500,
        'actor_lr': 1e-3,
        'value_lr': 3e-3,
        'batch_size': 32,
        'buffer_size': 45,
        'device': 'cpu',
    }
    # 4:30 h = 270min = 16200s
    if not model_config['result_dir'].exists():
        model_config['result_dir'].mkdir(parents=True)
    

    start_time = perf_counter()
    env_config = json.loads(model_config['config_path'].read_text())
    train_model(model_config, env_config)
    

    cost_time = str(timedelta(seconds = perf_counter() - start_time))
    
    
    print(f"Run in {cost_time}.")