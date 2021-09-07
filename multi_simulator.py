import random
import numpy as np
import json
from numpy.lib.function_base import angle
import torch
import argparse

from time import perf_counter
from datetime import timedelta
from typing import Dict, final
from pathlib import Path
from torch.optim import Adam
from copy import deepcopy

from core.utils import write_multi_records
from core.core import Core
from env.multi_env import MultiTradingEnv
from env.rl_agent import BaseAgent

def train_model(train_config: Dict, env_config: Dict):
    if not train_config['result_dir'].exists():
        train_config['result_dir'].mkdir()
    train_output_dir = train_config['result_dir'] / 'train'
    eval_output_dir = train_config['result_dir'] / 'eval'

    random_seed = 9527
    np.random.seed(random_seed)
    random.seed(random_seed)

    # init env
    def init_env(config: Dict):
        config = deepcopy(config)
        config['Market']['Securities']['TSMC']['price'] = [random.gauss(100, 1) for i in range(100)]
        config['Market']['Securities']['TSMC']['volume'] = [int(random.gauss(100, 10)*10) for i in range(100)]
        config['Market']['Securities']['TSMC']['value'] = [random.gauss(100, 1) for i in range(100)]
        return Core(config, market_type="call")

    def get_state(core, agent_id):
        look_back = rl_agents[agent_id].look_back
        market_stats = core.get_parallel_env_state(look_back)
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


    # init agent parameters
    device = torch.device(train_config['device'])
    if train_config['resume']:
        resume_config_path = train_config['resume_model_dir'] / 'config.json'
        resume_config = json.loads(resume_config_path.read_text())
        env_config['Agent']['RLAgent'] = resume_config['Agent']['RLAgent']
        num_rl_agent = env_config['Agent']['RLAgent'][0]['number']
        look_backs = env_config['Agent']['RLAgent'][0]['obs']['look_backs']
        action_spaces = [(3, 9, 5) for i in range(num_rl_agent)]
        observation_spaces = [look_backs[i]*2 + 2 for i in range(num_rl_agent)]
        agents = [BaseAgent(observation_space = observation_spaces[i], action_space = action_spaces[i], device = device, look_back = look_backs[i]) for i in range(num_rl_agent)]
        resume_model_path = train_config['resume_model_dir'] / 'model.pkl'
        checkpoint = torch.load(resume_model_path)
        for i, agent in enumerate(agents):
            agent.rl.load_state_dict(checkpoint[f"base_{i}"])

    else:
        num_rl_agent = env_config['Agent']['RLAgent'][0]['number']
        min_look_back = env_config['Agent']['RLAgent'][0]['obs']['min_look_back']
        max_look_back = env_config['Agent']['RLAgent'][0]['obs']['max_look_back']
        look_backs = [random.randint(min_look_back, max_look_back) for i in range(num_rl_agent)]
        action_spaces = [(3, 9, 5) for i in range(num_rl_agent)]
        observation_spaces = [look_backs[i]*2 + 2 for i in range(num_rl_agent)]
        env_config['Agent']['RLAgent'][0]['obs']['look_backs'] = look_backs
        agents = [BaseAgent(observation_space = observation_spaces[i], action_space = action_spaces[i], device = device, look_back = look_backs[i]) for i in range(num_rl_agent)]

    # record the observation spaces of agents
    with open(train_config['result_dir'] / 'config.json', 'w') as fp:
        json.dump(env_config, fp)


    # init training parameters
    lr = train_config['lr']
    n_epochs = train_config['n_epochs']
    n_steps = train_config['n_steps']
    batch_size = train_config['batch_size']
    optimizers = [Adam(agents[i].get_parameters(), lr = lr) for i in range(num_rl_agent)]



    history = []
    rl_group_name = f"{env_config['Agent']['RLAgent'][0]['name']}_{env_config['Agent']['RLAgent'][0]['number']}"

    for t in range(n_epochs):
        train_env = init_env(env_config)
        agent_ids = train_env.multi_env_start(random_seed, rl_group_name)
        # shuffle agent_id
        random.shuffle(agent_ids)

        rl_agents = {agent_id: agent for agent_id, agent in zip(agent_ids, agents)}
        rl_optimizers = {agent_id: optimizers for agent_id, optimizers in zip(agent_ids, optimizers)}
        states = {agent_id: get_state(train_env, agent_id) for agent_id in agent_ids}
        rl_records = {agent_id: {'states': [], 'actions': [], 'rewards': []} for agent_id in agent_ids}

        for i in range(n_steps):
            actions = {agent_id: agent.forward(states[agent_id]) for agent_id, agent in rl_agents.items()}
            train_env.multi_env_step(actions)
            next_states = {agent_id: get_state(train_env, agent_id) for agent_id in agent_ids}
            action_status = {agent_id: train_env.get_rl_agent_status(agent_id) for agent_id in agent_ids}
            rewards = {agent_id: agent.calculate_reward(actions[agent_id], next_states[agent_id], action_status[agent_id]) for agent_id, agent in rl_agents.items()}
  
            if i > 0 and i % batch_size == 0:
                for agent_id, agent in rl_agents.items():
                    rl_optimizers[agent_id].zero_grad()
                    loss = agent.calculate_loss()
                    loss.backward()
                    rl_optimizers[agent_id].step()
            
            for agent_id in agent_ids:
                rl_records[agent_id]['states'].append(states[agent_id])
                rl_records[agent_id]['actions'].append(actions[agent_id])
                rl_records[agent_id]['rewards'].append(rewards[agent_id])
            
            states = next_states
            print(f"At: {i}, the market state is:\n{train_env.show_market_state()}\n")
        
        for agent_id, agent in rl_agents.items():
            # final_reward = 0
            # rl_records[agent_id]['state'].append(states[agent_id])
            # rl_records[agent_id]['action'].append(actions[agent_id])
            # rl_records[agent_id]['rewards'].append(final_reward)
            # clear agent
            del agent.states[:]
            agent.rl.clear_memory()


        orderbooks, agent_manager = train_env.multi_env_close()
        history.append({'eps': t, 'orderbooks': orderbooks, 'agent_manager': agent_manager, 'states': rl_records})

        
    write_multi_records(history, train_output_dir)
    
    # store the agents
    model_output_path = train_config['result_dir'] / 'model.pkl'
    state_dicts = {f"base_{i}": agent.rl.state_dict() for i, agent in enumerate(agents)}
    torch.save(state_dicts, model_output_path)
    



def validate(env_config, model, n_steps, output_dir):
    output_dir = output_dir / 'test'
    if not output_dir.exists():
        output_dir.mkdir()
    env = MultiTradingEnv(env_config)
    obs = env.reset()

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

    orderbooks, agent_manager, states = env.close()

    # metric: wealth, valid action, reward, order placement, turnover rate


if __name__=='__main__':
    training_cofig = {
        'config_path': Path("config/multi.json"),
        'result_dir': Path("simulation_result/multi/run1/"),
        'resume': False,
        'resume_model_dir': Path("simulation_result/multi/test/"),
        'train': True,
        'n_steps': 3200,
        'n_epochs': 3,
        'batch_size': 64,
        'test': False,
        'test_steps': 16200,
        'lr': 1e-4,
        'total_timesteps': 16200,
        'device': 'cuda'

    }
    # 4:30 h = 270min = 16200s
    if not training_cofig['result_dir'].exists():
        training_cofig['result_dir'].mkdir(parents=True)
    
    start_time = perf_counter()
    env_config = json.loads(training_cofig['config_path'].read_text())
    if training_cofig['train']:
        train_model(training_cofig, env_config)
    
    if training_cofig['test']:
        validate(env_config, training_cofig['test_steps'], training_cofig['result_dir'])

    cost_time = str(timedelta(seconds = perf_counter() - start_time))
    
    
    print(f"Run in {cost_time}.")
