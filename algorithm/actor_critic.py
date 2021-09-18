import torch
import torch.nn.functional as F

from torch import nn
from torch import distributions
from torch.distributions import Categorical
from torch.optim import Adam
from collections import namedtuple


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, lr, batch_size, buffer_size, device):
        super(ActorCritic, self).__init__()
        self.action_space = action_space
        total_action_size = sum(action_space)

        self.affine = nn.Linear(observation_space, 32)
        self.action_layer = nn.Linear(32, total_action_size)
        self.value_layer = nn.Linear(32, 1)
        self.actor_optimizer = Adam(self.action_layer.parameters(), lr)
        self.value_optimizer = Adam(self.value_layer.parameters(), lr)

        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.gamma = 0.99


    def forward(self, state):
        with torch.no_grad():
            state = F.relu(self.affine(state))
            logits = self.action_layer(state).split(self.action_space)
        
        probs = [F.softmax(logit, dim = 0) for logit in logits]
        dists = [Categorical(prob) for prob in probs]
        actions = [dist.sample() for dist in dists]
        log_prob = sum([dist.log_prob(action) for dist, action in zip(dists, actions)])
        
        return [action.item() for action in actions], log_prob


    def get_batch_data(self):
        states, actions, rewards, log_probs, next_states = [], [], [], [], []
        for transition in self.buffer:
            states.append(transition.state)
            actions.append(transition.action)
            rewards.append(transition.reward)
            log_probs.append(transition.log_prob)
            next_states.append(transition.next_state)
        
        return [torch.tensor(item, dtype = torch.float, device = self.device) for item in [states, actions, rewards, log_probs, next_states]]

    def update(self):
        states, actions, rewards, log_probs, next_states = self.get_batch_data()
        states_value = self.value_layer(F.relu(self.affine(states)))
        next_states_value = self.value_layer(F.relu(self.affine(next_states)))
        
        # advantage & returns
        gae = 0
        advantages = []
        for step in reversed(range(self.buffer_size)):
            delta = rewards[step] + self.gamma * next_states_value[step] - states_value[step]
            gae = delta + self.gamma  * gae
            advantages.insert(0, gae)
        advantages = torch.stack(advantages, dim = 0)
        returns = advantages + states_value

        # normalize
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        action_loss = -log_probs * advantages
        value_loss = F.smooth_l1_loss(states_value, returns)
        loss = action_loss.mean() + value_loss.mean()
        
        # clear data
        del self.buffer[:]

        return loss

    
    def predict(self, state):
        state = F.relu(self.affine(state))
        
        logits = self.action_layer(state).split(self.action_space)
        probs = [F.softmax(logit, dim = 0) for logit in logits]
        actions = [probs[0].argmax() for prob in probs]
        
        return [action.item() for action in actions]