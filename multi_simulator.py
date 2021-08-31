import random
import numpy as np
import json
import argparse
import torch

from time import perf_counter
from datetime import timedelta
from typing import Dict
from pathlib import Path
from torch.optim import Adam

from env.utils import write_rl_records
from env.multi_env import MultiTradingEnv
from env.actor_critic import ActorCritic

def train_model(train_config: Dict, env_config: Dict):
    if not train_config['result_dir'].exists():
        train_config['result_dir'].mkdir()
    train_output_dir = train_config['result_dir'] / 'train'
    eval_output_dir = train_config['result_dir'] / 'eval'
    with open(train_config['result_dir'] / 'config.json', 'w') as fp:
        json.dump(env_config, fp)

    random_seed = 9527
    np.random.seed(random_seed)
    random.seed(random_seed)

    num_rl_agent = env_config['Agent']['RLAgent'][0]['number']
    lr = train_config['lr']
    n_epochs = train_config['n_epochs']
    n_steps = train_config['n_steps']
    batch_size = train_config['batch_size']
    device = torch.device(train_config['device'])


    train_env = MultiTradingEnv(env_config)
    look_backs = [random.randint(10, 20) for i in range(num_rl_agent)]
    action_spaces = [(3, 9, 5) for i in range(num_rl_agent)]
    agent_ids, observation_spaces, states = train_env.reset(look_backs)
    agents = {}
    optimizers = {}
    for i, agent_id in enumerate(agent_ids):
        agent = ActorCritic(obs_size = observation_spaces[i], action_shape = action_spaces[i])
        agents[agent_id] = agent.to(device)
        optimizers = {agent_id: Adam(agent.parameters(), lr = lr) for agent_id, agent in agents.items()}

    actions = {}
    '''
    make this shit run
    '''

    for t in range(n_epochs):
        for i in range(n_steps):
            for agent_id, agent in agents.items():
                state = torch.from_numpy(states[agent_id]).to(device)
                actions[agent_id] = agent.forward(state)
            states, rewards = train_env.step(actions)
            for agent_id, agent in agents.items():
                # reward = torch.tensor(rewards[agent_id], device = device)
                agent.rewards.append(rewards[agent_id])
    
            if i > 0 and i % batch_size == 0:
                for agent_id, agent in agents.items():
                    optimizers[agent_id].zero_grad()
                    loss = agent.calculate_loss()
                    loss.backward()
                    optimizers[agent_id].step()
                    agent.clear_memory()
            
            train_env.render()
        orderbooks, agent_manager, states = train_env.close()
        write_rl_records(orderbooks, agent_manager, states)
    
    



def test(env_config, model, n_steps, output_dir):
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

    write_rl_records(orderbooks, agent_manager, states, output_dir)

if __name__=='__main__':
    training_cofig = {
        'config_path': Path("config/multi.json"),
        'result_dir': Path("simulation_result/multi/te/"),
        'resume': False,
        'resume_model_dir': Path("rl_result/noir_10300/model.zip"),
        'train': True,
        'n_steps': 18000,
        'n_epochs': 1,
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
    
    # if training_cofig['test']:
    #     test(env_config, model, training_cofig['test_steps'], training_cofig['result_dir'])

    # model = ppo.PPO.load(model_dir)
    cost_time = str(timedelta(seconds = perf_counter() - start_time))
    
    
    print(f"Run in {cost_time}.")
