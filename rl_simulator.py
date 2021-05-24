import random
import numpy as np
import json
import argparse
from time import perf_counter
from datetime import timedelta
from pathlib import Path
from stable_baselines3.ppo import ppo
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from typing import Dict

from env.utils import write_rl_records
from env.rl_agent import FeatureExtractor
from env.trading_env import TradingEnv
from env.utils import TrainingInfoCallback

def train_model(config: Dict, output_dir: Path):
    if not output_dir.exists():
        output_dir.mkdir()
    with open(output_dir / 'config.json', 'w') as fp:
        json.dump(config, fp)

    random_seed = 9527
    np.random.seed(random_seed)
    random.seed(random_seed)
    trading_env = TradingEnv(config)
    # trading_env = Monitor(trading_env)


    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
    )
    model = ppo.PPO(policy="MultiInputPolicy",
                    env = trading_env,
                    n_steps = 1024,
                    verbose = 1,
                    policy_kwargs = policy_kwargs)
    callback = TrainingInfoCallback(check_freq = 1000, result_dir = output_dir)
    model.learn(total_timesteps=20480, callback=callback)
    model.save(output_dir / "timestep_10300")
    return model

def evaluate(config, model, output_dir):
    output_dir = output_dir / 'eval'
    if not output_dir.exists():
        output_dir.mkdir()
    env = TradingEnv(config)
    obs = env.reset()
    n_steps = 5000
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

    orderbooks, agent_manager, states = env.close()

    # metric: wealth, valid action, reward, order placement, turnover rate

    write_rl_records(orderbooks, agent_manager, states, output_dir)

if __name__=='__main__':
    config_path = Path("config/config_zi.json")
    # model_dir = Path("trained_model/timestep_10300")
    result_dir = Path("rl_result/timestep_10300/")
    if not result_dir.exists():
        result_dir.mkdir(parents=True)
    
    start_time = perf_counter()
    config = json.loads(config_path.read_text())
    model = train_model(config, result_dir)
    # model = ppo.PPO.load(result_dir / "timestep_5000")
    cost_time = str(timedelta(seconds = perf_counter() - start_time))
    evaluate(config, model, result_dir)
    
    print(f"Run in {cost_time}.")

    