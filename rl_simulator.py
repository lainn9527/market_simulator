import random
import numpy as np
import json
import argparse
from time import perf_counter
from datetime import timedelta
from typing import Dict
from pathlib import Path
from stable_baselines3.ppo import ppo
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from env.utils import write_rl_records
from env.rl_agent import FeatureExtractor
from env.trading_env import TradingEnv
from env.utils import TrainingInfoCallback, evaluate

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
    trading_env = TradingEnv(env_config)
    eval_env = TradingEnv(env_config)
    # trading_env = Monitor(trading_env)


    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
    )
    '''
    n_steps: collect n steps in the buffer and update
    batch_size: number of timesteps in every update
    n_epochs: number of times of using the same buffer to update

    in model.learn()
    total_timesteps: total number of steps to train

    For example in the following setting:
        n_steps = 1024,
        batch_size = 64,
        n_epochs = 10,
        and total_timesteps = 10240
    then in every 1024 steps, the model will train on batch of size 64 for 10 times and the total steps are 10240.
    So the model will be updated for (10240 / 1024) * (1024 / 64) * 10 = 1600 times
    '''
    if train_config['resume']:
        model = ppo.PPO.load(train_config['resume_model_dir'])
    else:
        model = ppo.PPO(policy = "MultiInputPolicy",
                        env = trading_env,
                        learning_rate = train_config['learning_rate'],
                        n_steps = train_config['n_steps'],
                        batch_size = train_config['batch_size'],
                        n_epochs = train_config['n_epochs'],
                        verbose = 1,
                        tensorboard_log = train_output_dir,
                        policy_kwargs = policy_kwargs)
    # down tuan
    callback = TrainingInfoCallback(check_freq = train_config['n_steps'],
                                    result_dir = train_output_dir)
    
    model.learn(total_timesteps = train_config['total_timesteps'],
                callback = callback)
    
    model.save(train_config['result_dir'] / "model")
    return model

def test(env_config, model, n_steps, output_dir):
    output_dir = output_dir / 'test'
    if not output_dir.exists():
        output_dir.mkdir()
    env = TradingEnv(env_config)
    obs = env.reset()

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

    orderbooks, agent_manager, states = env.close()

    # metric: wealth, valid action, reward, order placement, turnover rate

    write_rl_records(orderbooks, agent_manager, states, output_dir)

if __name__=='__main__':
    training_cofig = {
        'config_path': Path("config/config_zi.json"),
        'result_dir': Path("rl_result/te/"),
        'resume': False,
        'resume_model_dir': Path("rl_result/noir_10300/model.zip"),
        'train': True,
        'test': False,
        'test_steps': 16200,
        'learning_rate': 1e-4,
        'n_steps': 1800,
        'batch_size': 60,
        'n_epochs': 1,
        'total_timesteps': 16200,

    }
    # 4:30 h = 270min = 16200s
    # 30m = 1800
    if not training_cofig['result_dir'].exists():
        training_cofig['result_dir'].mkdir(parents=True)
    
    start_time = perf_counter()
    env_config = json.loads(training_cofig['config_path'].read_text())
    if training_cofig['train']:
        model = train_model(training_cofig, env_config)
    
    if training_cofig['test']:
        test(env_config, model, training_cofig['test_steps'], training_cofig['result_dir'])

    # model = ppo.PPO.load(model_dir)
    cost_time = str(timedelta(seconds = perf_counter() - start_time))
    
    
    print(f"Run in {cost_time}.")
