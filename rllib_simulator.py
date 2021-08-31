import os
import ray
import json
import torch
from torch import nn
from copy import deepcopy
from numpy import float32
from supersuit import normalize_obs_v0, dtype_v0, color_reduction_v0
from ray.rllib.agents import ppo
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork 
from ray.tune.registry import register_env
from pathlib import Path
from time import perf_counter

from env.parallel_trading_env import ParallelTradingEnv



class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = FullyConnectedNetwork(obs_space, action_space, num_outputs,
                                                     model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])

if __name__ == "__main__":
    training_cofig = {
        'config_path': Path("config/call.json"),
        'result_dir': Path("parallel/te/"),
        'resume': False,
        'resume_model_dir': Path("rl_result/noir_10300/model.zip"),
        'train': True,
        'test': False,
        'test_steps': 1000,
        'learning_rate': 1e-4,
        'n_steps': 1000,
        'batch_size': 60,
        'n_epochs': 1,
        'total_timesteps': 1000,

    }
    # 4:30 h = 270min = 16200s
    # 30m = 1800
    if not training_cofig['result_dir'].exists():
        training_cofig['result_dir'].mkdir(parents=True)
    
    start_time = perf_counter()
    env_config = json.loads(training_cofig['config_path'].read_text())

    alg_name = "PPO"

    # Function that outputs the environment you wish to register.
    def env_creator(config):
        print(f"The env config is : {config}")
        env = ParallelTradingEnv(config)
        return env

    num_cpus = 4
    num_rollouts = 2

    # Gets default training configuration and specifies the POMgame to load.
    config = ppo.DEFAULT_CONFIG.copy()


    # Register env
    register_env("trading",
                 lambda config: ParallelPettingZooEnv(env_creator(config)))
    env = ParallelPettingZooEnv(env_creator(env_config))
    observation_space = env.observation_space
    action_space = env.action_space
    del env

    # Register model
    ModelCatalog.register_custom_model("my_model", TorchCustomModel)

    # Configuration for multiagent setup with policy sharing:
    # config['framework'] = 'torch'
    config["multiagent"] = {
        # Setup a single, shared policy for all agents.
        "policies": {
            "ppo_policy": (ppo.PPOTorchPolicy, observation_space, action_space, {})
        },
        # Map all agents to that policy.
        "policy_mapping_fn": lambda agent_id, episode, **kwargs: "ppo_policy",
        "policies_to_train": ["ppo_policy"],

    }

    # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
    config["num_gpus"] = 1
    config["log_level"] = "DEBUG"
    config["num_workers"] = 1
    # Fragment length, collected at once from each worker and for each agent!
    config["rollout_fragment_length"] = 30
    # Training batch size -> Fragments are concatenated up to this point.
    config["train_batch_size"] = 200
    # After n steps, force reset simulation
    config["horizon"] = 1000
    # Default: False
    config["no_done_at_end"] = False
    # Info: If False, each agents trajectory is expected to have
    # maximum one done=True in the last step of the trajectory.
    # If no_done_at_end = True, environment is not resetted
    # when dones[__all__]= True.
    config['env_config'] = env_config

    # Initialize ray and trainer object
    ray.init(num_cpus=num_cpus, num_gpus=1, local_mode = False)
    trainer = ppo.PPOTrainer(env="trading", config=config)

    # Train once
    result = trainer.train()
    print(result)