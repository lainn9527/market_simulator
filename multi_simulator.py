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
from multiprocessing import Pool

from core.utils import write_multi_records
from core.multi_env import MultiTradingEnv

Transition = namedtuple('Transition',['state', 'action', 'reward', 'log_prob', 'next_state', 'done'])
def train_model(train_config: Dict):
    if not train_config['result_dir'].exists():
        train_config['result_dir'].mkdir()
    train_output_dir = train_config['result_dir'] / 'train'
    validate_output_dir = train_config['result_dir'] / 'validate'
    model_output_dir = train_config['result_dir'] / 'model'
    model_output_dir.mkdir(exist_ok = True, parents = True)

    random_seed = 9528
    np.random.seed(random_seed)
    random.seed(random_seed)

    # check if resume
    if train_config['resume']:
        resume_config_path = train_config['resume_model_dir'] / 'config.json'
        env_config = json.loads(resume_config_path.read_text())
    else:
        env_config = json.loads(train_config['config_path'].read_text())

    # init training env
    multi_env = MultiTradingEnv()
    rl_agent_config = env_config['Agent']['RLAgent']
    agents = multi_env.build_agents(rl_agent_config,
                                    train_config['actor_lr'],
                                    train_config['value_lr'],
                                    train_config['batch_size'],
                                    train_config['buffer_size'],
                                    train_config['n_epoch'],
                                    train_config['device'],
                                    train_config['resume'],
                                    train_config['resume_model_dir']
                                )
        

    # record the observation spaces of agents
    with open(train_config['result_dir'] / 'config.json', 'w') as fp:
        json.dump(env_config, fp)
        print(f"Store config in {train_config['result_dir'] / 'config.json'}")
        
    # init training parameters
    print("Start training...")
    n_epochs = train_config['train_epochs']
    train_steps = train_config['train_steps']
    validate_steps = train_config['validate_steps']
    early_stop_patience = train_config['early_stop_patience']
    earlt_stop_threshold = train_config['earlt_stop_threshold']

    trained_agent_ids = []
    for t in range(n_epochs):
        print(f"Epoch {t} start")
        agent_ids, states = multi_env.reset(env_config)
        training_agent_ids = {agent_id: True for agent_id in agent_ids}
        for trained_agent_id in trained_agent_ids:
            training_agent_ids[trained_agent_id] = False
            

        rl_records = {agent_id: {'states': [], 'actions': [], 'rewards': [], 'policy_loss': [], 'value_loss': []} for agent_id in agent_ids}
        for i in range(train_steps):
            # collect actions
            obs, actions, log_probs, rewards = {}, {}, {}, {}
            for agent_id, agent in zip(agent_ids, agents):
                obs[agent_id], actions[agent_id], log_probs[agent_id] = agent.get_action(states[agent_id])

            # rewards, next_steps
            done, rewards, next_states, next_obs = multi_env.step(actions)

            for agent_id, agent in zip(agent_ids, agents):
                if not training_agent_ids[agent_id]:
                    continue
                loss = agent.update(Transition(obs[agent_id],actions[agent_id], rewards[agent_id]['weighted_reward'], log_probs[agent_id], next_obs[agent_id], done))

                # log
                rl_records[agent_id]['states'].append({'observations': obs[agent_id].tolist(), 'agent_state': states[agent_id]['agent']})
                rl_records[agent_id]['actions'].append(actions[agent_id])
                rl_records[agent_id]['rewards'].append(rewards[agent_id])
                
                if loss != None:
                    rl_records[agent_id]['policy_loss'] += [loss['policy_loss']]
                    rl_records[agent_id]['value_loss'] += [loss['value_loss']]
                    # check stop training
                    if len(rl_records[agent_id]['policy_loss']) >= early_stop_patience:
                        policy_array = np.array(rl_records[agent_id]['policy_loss'][-early_stop_patience:])
                        value_array = np.array(rl_records[agent_id]['value_loss'][-early_stop_patience:])
                        policy_stop = ((( policy_array[:-1] - policy_array[1:] ) / np.abs(policy_array[:-1])) < earlt_stop_threshold).all()
                        value_stop = ((( value_array[:-1] - value_array[1:] ) / np.abs(value_array[:-1])) < earlt_stop_threshold).all()
                        if policy_stop and value_stop:
                            trained_agent_ids.append(agent_id)
                            training_agent_ids[agent_id] = False
                            print(f'{agent_id} stop training, remain {len(agent_ids) - len(trained_agent_ids)}')

            
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
        multi_env.store_agents(model_output_dir, env_config)
        

        # validate
        if train_config['validate']:
            print("Start validating...")
            agent_ids, states = multi_env.reset(env_config)
            rl_records = {agent_id: {'states': [], 'actions': [], 'rewards': []} for agent_id in agent_ids}

            for i in range(validate_steps):
                # collect actions
                obs, actions = {}, {}
                for agent_id, agent in zip(agent_ids, agents):
                    obs[agent_id], actions[agent_id], _ = agent.get_action(states[agent_id])

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


        if len(trained_agent_ids) == len(agent_ids):
            print('All agents are trained. Stop training.')
            return
        else:
            print(f"{len(trained_agent_ids)} agents have stopped.")




def predict_model(train_config: Dict, n_parallel = 1):
    predict_output_dir = train_config['result_dir'] / 'predict'

    random_seed = 9528
    np.random.seed(random_seed)
    random.seed(random_seed)

    if train_config['train']:
        resume_config_path = train_config['result_dir'] / 'config.json'
        resume_model_dir = train_config['result_dir'] / 'model'
    elif train_config['resume']:
        resume_config_path = train_config['resume_model_dir'] / 'config.json'
        resume_model_dir = train_config['resume_model_dir']
    else:
        raise Exception('No model to predict.')

    env_config = json.loads(resume_config_path.read_text())

    # init training env
    multi_env = MultiTradingEnv()
    rl_agent_config = env_config['Agent']['RLAgent']
    agents = multi_env.build_agents(rl_agent_config,
                                    train_config['actor_lr'],
                                    train_config['value_lr'],
                                    train_config['batch_size'],
                                    train_config['buffer_size'],
                                    train_config['n_epoch'],
                                    train_config['device'],
                                    resume = True,
                                    resume_model_dir = resume_model_dir
                                )
        

    print("Start predict...")
    n_epochs = train_config['predict_epochs']
    n_steps = train_config['predict_steps']
    args_list = [(predict_output_dir / f"sim_{i}", n_steps, agents, multi_env, env_config, random_seed+i, ) for i in range(n_epochs)]
    # env_config['Agent'].pop('ScalingAgent')

    # predict(predict_output_dir / f"sim_{0}", n_steps, agents, multi_env, env_config, random_seed, )
    with Pool(n_parallel) as p:
        p.starmap(predict, args_list)



def predict(output_dir: Path, n_steps, agents, multi_env, env_config, random_seed = 9528):
    np.random.seed(random_seed)
    random.seed(random_seed)

    agent_ids, states = multi_env.reset(env_config)
    rl_records = {agent_id: {'states': [], 'actions': [], 'rewards': []} for agent_id in agent_ids}
    number = output_dir.parts[-1][-1]
    print(f"Prediction {number} start")
    for i in range(n_steps):
        # collect actions
        obs, actions = {}, {}
        for agent_id, agent in zip(agent_ids, agents):
            obs[agent_id], actions[agent_id], _ = agent.get_action(states[agent_id])

        # rewards, next_steps
        done, rewards, next_states, next_obs = multi_env.step(actions)

        for agent_id, agent in zip(agent_ids, agents):
            # log
            rl_records[agent_id]['states'].append({'observations': obs[agent_id].tolist(), 'agent_state': states[agent_id]['agent']})
            rl_records[agent_id]['actions'].append(actions[agent_id])
        
        states = next_states
        # if i % 10 == 0:
        #     multi_env.render(i)

        if done:
            multi_env.render(i)
            print(f"No quote. End of this episode")
            break    
    
    for agent_id, agent in zip(agent_ids, agents):
        agent.end_episode()

    orderbooks, agent_manager = multi_env.close()
    predict_record = {'orderbooks': orderbooks, 'agent_manager': agent_manager, 'states': rl_records}
    write_multi_records(predict_record, output_dir)    
    print(f"Prediction result is stored in {output_dir}")

if __name__=='__main__':
    experiment_name = 'multi'
    config_name = 'sc_100'
    model_config = {
        'config_path': Path(f"config/{experiment_name}/{config_name}.json"),
        'result_dir': Path(f"simulation_result/experiment/{experiment_name}/{config_name}"),
        # 'result_dir': Path(f"simulation_result/experiment/{experiment_name}/{config_name}_bf45/"),
        'resume': False,
        'resume_model_dir': Path(f"simulation_result/experiment/{experiment_name}/{config_name}_/model"),
        # 'resume_model_dir': Path(f"model/sc_100_bs4_br8_wealthreward"),
        'train': True,
        'train_epochs': 10,
        'train_steps': 500,
        'validate': False,
        'validate_steps': 200,
        'predict': True,
        'predict_epochs': 6,
        'predict_steps': 500,
        'actor_lr': 1e-3,
        'value_lr': 3e-3,
        'batch_size': 4,
        'buffer_size': 8,
        'n_epoch': 10,
        'early_stop_patience': 10,
        'earlt_stop_threshold': 0.05,
        'device': 'cpu',
    }
    # 4:30 h = 270min = 16200s
    if not model_config['result_dir'].exists():
        model_config['result_dir'].mkdir(parents=True)
    

    start_time = perf_counter()

    if model_config['train']:
        train_model(model_config)
    else:
        print('Skip training.')

    if model_config['predict']:
        predict_model(model_config)
    else:
        print('Skip predicting')

    print("End the simulation")

    cost_time = str(timedelta(seconds = perf_counter() - start_time))
    
    
    print(f"Run in {cost_time}.")