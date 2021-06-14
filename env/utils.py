import os
import gym
import json
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict
from pathlib import Path
from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import JSONOutputFormat

from core.utils import write_records
from env.trading_env import TradingEnv

class TrainingInfoCallback(BaseCallback):
    def __init__(self,
                 check_freq: int,
                 result_dir: Path,
                #  eval_env: TradingEnv,
                #  eval_freq: int,
                #  eval_result_dir: Path,
                 verbose=1):
        super(TrainingInfoCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.result_dir = result_dir
        # self.eval_env = eval_env
        # self.eval_freq = eval_freq
        # self.eval_result_dir = eval_result_dir

        result_dir.mkdir(exist_ok=True)
        # eval_result_dir.mkdir()
        # self.log_dir = log_dir
        # self.save_path = os.path.join(log_dir, 'best_model')
        # self.best_mean_reward = -np.inf

        # def _init_callback(self) -> None:
        #     # Create folder if needed
        #     if self.save_path is not None:
        #         os.makedirs(self.save_path, exist_ok=True)

    def _init_callback(self) -> None:
        self.training_env = self.training_env.envs[0]

    def _on_step(self) -> bool:
        return True
        timestep = self.training_env.core.timestep
        if timestep % self.check_freq != 0:
            return
        
        record_states = self.training_env.states[-1*(self.check_freq):]
        start_state = self.training_env.start_state
        evaluate(record_states, start_state)
        return True

    def _on_rollout_end(self) -> None:
        self.logger
        record_states = self.training_env.states[-1*(self.check_freq):]
        start_state = self.training_env.start_state
        valid_action_rate, wealth_reward, action_reward, wealth_gain, wealth_gain_from_start = evaluate(record_states, start_state)
        self.logger.record("train/valid_action_rate", valid_action_rate)
        self.logger.record("train/wealth_reward", wealth_reward)
        self.logger.record("train/action_reward", action_reward)
        self.logger.record("train/wealth_gain", wealth_gain)
        self.logger.record("train/wealth_gain_from_start", wealth_gain_from_start)
        
        return

    def _on_training_end(self) -> None:
        evaluate(self.training_env.states, self.training_env.start_state)
        # since the trading_env is <Monitor> class and it has Monitor too, we need to grad the our env
        orderbooks, agent_manager, states = self.training_env.env.close()
        write_rl_records(orderbooks, agent_manager, states, self.result_dir)
    

def evaluate(states: Dict, start_state):
    # return metrics
    if len(states) == 0:
        return

    total_reward = 0
    action_reward = 0
    wealth_reward = 0
    valid_actions = 0
    valid_orders = 0

    # there is no reward and action in the latest state
    for state in states[:-1]:
        total_reward += state['reward']['total_reward']
        action_reward += state['reward']['action_reward']
        wealth_reward += state['reward']['wealth_reward']

        valid_actions += state['action']['is_valid']
        if (state['action']['action'][0] == 1 or state['action']['action'][0] == 2) and state['action']['is_valid']:
            valid_orders += 1
    total_reward = round(total_reward/len(states), 2)
    action_reward = round(action_reward/len(states), 2)
    wealth_reward = round(wealth_reward/len(states), 2)

    valid_action_rate = round(valid_actions / len(states), 2)
    wealth_gain = round((states[-1]['agent']['wealth'] - states[0]['agent']['wealth'])/states[0]['agent']['wealth'], 4)
    wealth_gain_from_start = round( (states[-1]['agent']['wealth'] - start_state['agent']['wealth'])/start_state['agent']['wealth'], 4)
    print(f"At: {states[-1]['timestep']}, the close price is:\n{states[-1]['price']['close'][-1]}")
    print(f'Action: valid order={valid_orders}, valid action rate={valid_action_rate}')
    print(f'Reward: total={total_reward}, wealth={wealth_reward}, action={action_reward}')
    print(f"Wealth: {wealth_gain * 100}% in {len(states)}s, {wealth_gain_from_start * 100}% from start\n" )
    
    return valid_action_rate, wealth_reward, action_reward, round(wealth_gain * 100, 2), round(wealth_gain_from_start * 100, 2)

def write_rl_records(orderbooks: Dict, agent_manager, states: Dict, output_dir: Path):
    write_records(orderbooks, agent_manager, output_dir)
    
    rl_id = "RL"
    rl_agent = agent_manager.agents[rl_id]
    order_ids = rl_agent.orders['TSMC'] + rl_agent.orders_history['TSMC']

    # output the order of rl agent
    agent_orders = []
    for order_id in order_ids:
        order_record = orderbooks['TSMC'].orders[order_id]
        order = order_record.order
        agent_orders.append({'time': int(order_record.placed_time), 'bid_or_ask': str(order.bid_or_ask), 'price': float(order.price), 'volume': int(order.quantity)})
        # order_price = order.price
        # state = states[order_record.placed_time]
        # close_price = state['price']['close']
        # slippage = close_price - order_price

    file_path = output_dir / "rl.json"


    with open(file_path, 'w') as fp:
        json.dump({'orders' :agent_orders, 'states': states}, fp, indent=4)
        