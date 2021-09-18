from numpy.core.fromnumeric import clip
import torch
import torch.nn.functional as F
from torch import nn
from torch import distributions
from torch.distributions import Categorical
from torch.optim import Adam


class PPO(nn.Module):
    def __init__(self, observation_space, action_space, lr, batch_size, buffer_size, device):
        super(PPO, self).__init__()
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
        self.clip_range = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 1

    def forward(self, state):
        with torch.no_grad():
            state = F.relu(self.affine(state))
            logits = self.action_layer(state).split(self.action_space)
        
        probs = [F.softmax(logit, dim = 0) for logit in logits]
        dists = [Categorical(prob) for prob in probs]
        actions = [dist.sample() for dist in dists]
        log_prob = sum([dist.log_prob(action) for dist, action in zip(dists, actions)])
        
        return [action.item() for action in actions], log_prob


    def get_buffer_data(self):
        states, actions, rewards, log_probs, next_states = [], [], [], [], []
        for transition in self.buffer:
            states.append(transition.state)
            actions.append(transition.action)
            rewards.append(transition.reward)
            log_probs.append(transition.log_prob)
            next_states.append(transition.next_state)
        
        return [torch.tensor(item, dtype = torch.float, device = self.device) for item in [states, actions, rewards, log_probs, next_states]]

    def get_mini_batch(self):
        pass

    def update(self):
        states, actions, rewards, log_probs, next_states = self.get_buffer_data()
        values = self.value_layer(F.relu(self.affine(states))).detach().squeeze()
        next_values = self.value_layer(F.relu(self.affine(next_states))).detach().squeeze()
        

        # advantage & returns
        gae = 0
        advantages = []
        for step in reversed(range(self.buffer_size)):
            delta = rewards[step] + self.gamma * next_values[step] - values[step]
            gae = delta + self.gamma  * gae
            advantages.insert(0, gae)
        advantages = torch.stack(advantages, dim = 0)
        returns = advantages + values

        start_idx = 0
        # normalize
        while start_idx < self.buffer_size:
            end_idx = start_idx + self.batch_size if (start_idx + self.batch_size) <= self.buffer_size else self.buffer_size
            batch_returns = returns[start_idx:end_idx]
            batch_states = states[start_idx:end_idx]
            batch_actions = actions[start_idx:end_idx]
            batch_advantages = advantages[start_idx:end_idx]
            old_values = values[start_idx: end_idx]
            old_log_probs = log_probs[start_idx:end_idx]
            clip_range = self.clip_range
            
            # normalize
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
            batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std() + 1e-8)

            # policy loss
            logits = self.action_layer(F.relu(self.affine(batch_states))).split(self.action_space, dim = 1)
            probs = [F.softmax(logit, dim = 1) for logit in logits]
            dists = [Categorical(prob) for prob in probs]
            new_log_probs = dists[0].log_prob(batch_actions[:, 0]) + dists[1].log_prob(batch_actions[:, 1]) + dists[2].log_prob(batch_actions[:, 2])
            ratio = torch.exp(new_log_probs - old_log_probs)
            policy_loss_1 = batch_advantages * ratio
            policy_loss_2 = batch_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

            # entropy
            entropy_loss = (dists[0].entropy().mean() + dists[1].entropy().mean() + dists[2].entropy().mean()) / 3

            # value loss
            new_values = self.value_layer(F.relu(self.affine(batch_states))).squeeze()
            value_pred = torch.clamp(new_values, old_values - clip_range, old_values + clip_range)
            value_loss_1 = F.mse_loss(value_pred, batch_returns)
            value_loss_2 = F.mse_loss(new_values, batch_returns)
            value_loss = torch.max(value_loss_1, value_loss_2)

            # total loss
            loss = policy_loss + self.entropy_coef*entropy_loss + self.value_coef*value_loss 

            # optimize
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
            start_idx = end_idx

        # clear data
        del self.buffer[:]


    
    def predict(self, state):
        with torch.no_grad():
            state = F.relu(self.affine(state))
            logits = self.action_layer(state).split(self.action_space)
        
        probs = [F.softmax(logit, dim = 0) for logit in logits]
        actions = [probs[0].argmax() for prob in probs]
        
        return [action.item() for action in actions]