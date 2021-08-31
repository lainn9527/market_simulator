
import gym
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class FeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(FeatureExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        '''
        spaces.Dict({
            'orderbook': spaces.Box(low = 0, high = 1000, shape=(2, self.config['obs']['best_price'], )),
            'price': spaces.Box(low = 0, high = 2000, shape=(5, self.config['obs']['lookback'],)),
            'agent': spaces.Box(low = 0, high = 100000000, shape=(2,))
        })
        '''


        extractors['orderbook'] = nn.GRU(input_size = observation_space.spaces['orderbook'].shape[0],
                                          num_layers = 1,
                                          hidden_size = 2,
                                          batch_first = True)

        extractors['price'] = nn.GRU(input_size = observation_space.spaces['price'].shape[0],
                                     num_layers = 2,
                                     hidden_size = 8,
                                     batch_first = True)

        extractors['agent'] = nn.Linear(in_features = observation_space.spaces['agent'].shape[0],
                                        out_features = 1)

        total_concat_size = 2 + 8 + 1

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == 'agent':
                encoded_tensor_list.append(extractor(observations[key]))
            else:
                # input size: (batch, feature, seq_len) to (batch, seq_len, feature) by transpose
                output, _ = extractor(observations[key].transpose(1, 2))
                # output size: (batch, seq_len, hidden_size), extract the last sequence output
                encoded_tensor_list.append(output[:, -1, :])
        return th.cat(encoded_tensor_list, dim=1)

class ParallelFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(ParallelFeatureExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        '''
        spaces.Dict({
            'orderbook': spaces.Box(low = 0, high = 1000, shape=(2, self.config['obs']['best_price'], )),
            'price': spaces.Box(low = 0, high = 2000, shape=(5, self.config['obs']['lookback'],)),
            'agent': spaces.Box(low = 0, high = 100000000, shape=(2,))
        })
        '''


        self.extractors = nn.Sequential(nn.Linear(in_features = observation_space.shape[0], out_features = 8),
                                        nn.Linear(in_features = 8, out_features = 2),)
        self._features_dim = 2

    def forward(self, observations) -> th.Tensor:
        
        return self.extractors(observations)