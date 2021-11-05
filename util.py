import json
import torch
import numpy as np
import random

from pathlib import Path
from algorithm.ppo import PPO

n_agent = 100
min_look_back = 2
max_look_back = 100

# resume_model_dir = Path('simulation_result/experiment/add_rl_in_classic/va_1_wealth/model')
# resume_model_path = resume_model_dir / 'model.pkl'
# model_output_dir = Path(f'model/va_{n_agent}_trained_actions/')
# model_output_dir.mkdir(exist_ok=True, parents=True)

# model_output_path = model_output_dir / 'model.pkl'
# checkpoint = torch.load(resume_model_path)
# record = checkpoint['ppo_va_1_0']
# output = {}
# for i in range(n_agent):
#     agent_id = f'ppo_va_{n_agent}_{i}'
#     output[agent_id] = record

# torch.save(output, model_output_path)

# rand_weight = np.random.rand(n_agent, 4)
# wealth_weight = rand_weight / rand_weight.sum(1).reshape([-1, 1])
# wealth_weight = wealth_weight.tolist()
# look_backs = [random.randint(min_look_back, max_look_back) for i in range(n_agent)]

# config_path = Path('config/add_rl_in_classic/va_100.json')
# env_config = json.loads(config_path.read_text())
# env_config['Agent']['RLAgent'][0]['look_backs'] = look_backs
# env_config['Agent']['RLAgent'][0]['weight_wealth'] = wealth_weight

# with open(model_output_dir / 'config.json', 'w') as fp:
#     json.dump(env_config, fp)

model_output_path = 'model/va_100_trained_actions/model.pkl'
config_path = Path('model/va_100_trained_actions/config.json')

va_check = torch.load(model_output_path)
va_config = json.loads(config_path.read_text())

model_output_path = 'model/tr_100_trained_actions/model.pkl'
config_path = Path('model/tr_100_trained_actions/config.json')

tr_check = torch.load(model_output_path)
tr_config = json.loads(config_path.read_text())

model_output_path = 'model/sc_100_bs4_br8_wealthreward/model.pkl'
config_path = Path('model/sc_100_bs4_br8_wealthreward/config.json')

sc_check = torch.load(model_output_path)
sc_config = json.loads(config_path.read_text())

sc_check