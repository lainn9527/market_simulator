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
    validate_output_dir = train_config['result_dir'] / 'validate'
    predict_output_dir = train_config['result_dir'] / 'predict'

    random_seed = 9528
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
        print(f"Store config in {train_config['result_dir'] / 'config.json'}")
        
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
                    rl_records[agent_id]['states'].append({'observations': obs[agent_id].tolist(), 'agent_state': states[agent_id]['agent']})
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

            # validate
            if train_config['validate']:
                print("Start validating...")
                n_steps = train_config['validate_steps']
                agent_ids, states = multi_env.reset(env_config)
                rl_records = {agent_id: {'states': [], 'actions': [], 'rewards': []} for agent_id in agent_ids}

                for i in range(n_steps):
                    # collect actions
                    obs, actions = {}, {}
                    for agent_id, agent in zip(agent_ids, agents):
                        obs[agent_id], actions[agent_id] = agent.predict(states[agent_id])

                    # rewards, next_steps
                    done, rewards, next_states, next_obs = multi_env.step(actions)

                    for agent_id, agent in zip(agent_ids, agents):
                        # log
                        rl_records[agent_id]['states'].append({'observations': obs[agent_id].tolist(), 'agent_state': states[agent_id]['agent']})
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
        print("Skip training")

    if train_config['predict']:
        print("Start predict...")
        # validate
        predict_output_dir = train_config['result_dir'] / 'predict'
        n_epochs = train_config['predict_epochs']
        n_steps = train_config['predict_steps']

        for t in range(n_epochs):
            print(f"Epoch {t} start")
            agent_ids, states = multi_env.reset(env_config)
            rl_records = {agent_id: {'states': [], 'actions': [], 'rewards': []} for agent_id in agent_ids}

            for i in range(n_steps):
                # collect actions
                obs, actions = {}, {}
                for agent_id, agent in zip(agent_ids, agents):
                    obs[agent_id], actions[agent_id] = agent.predict(states[agent_id])

                # rewards, next_steps
                done, rewards, next_states, next_obs = multi_env.step(actions)

                for agent_id, agent in zip(agent_ids, agents):
                    # log
                    rl_records[agent_id]['states'].append({'observations': obs[agent_id].tolist(), 'agent_state': states[agent_id]['agent']})
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
            predict_record = {'eps': t, 'orderbooks': orderbooks, 'agent_manager': agent_manager, 'states': rl_records}
            write_multi_records(predict_record, predict_output_dir / f'sim_{t}')    
            print(f"Prediction result is stored in {predict_output_dir / f'sim_{t}'}")

    else:
        print("Skip predicting")

    print("End the simulation")


if __name__=='__main__':
    config_name = 'all_250'
    model_config = {
        'config_path': Path(f"config/{config_name}.json"),
        'result_dir': Path(f"simulation_result/multi/{config_name}_entropy_diff_hyper/"),
        'resume': False,
        'resume_model_dir': Path("simulation_result/multi/scaling_250_pure_wealth/"),
        'train': True,
        'train_epochs': 2,
        'train_steps': 1000,
        'validate': True,
        'validate_steps': 1000,
        'predict': True,
        'predict_epochs': 5,
        'predict_steps': 1000,
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