import torch
import torch.nn.functional as F
from torch import nn
from torch import distributions
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCritic, self).__init__()
        self.action_space = action_space
        total_action_size = sum(action_space)

        self.affine = nn.Linear(observation_space, 32)
        self.action_layer = nn.Linear(32, total_action_size)
        self.value_layer = nn.Linear(32, 1)
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []

    def forward(self, state):
        state = F.relu(self.affine(state))
        state_value = self.value_layer(state)
        
        logits = self.action_layer(state).split(self.action_space)
        probs = [F.softmax(logit, dim = 0) for logit in logits]
        dists = [Categorical(prob) for prob in probs]
        actions = [dist.sample() for dist in dists]
        
        self.logprobs.append(sum([dist.log_prob(action) for dist, action in zip(dists, actions)]))
        self.state_values.append(state_value)
        
        return [action.item() for action in actions]
    
    def predict(self, state):
        state = F.relu(self.affine(state))
        state_value = self.value_layer(state)
        
        logits = self.action_layer(state).split(self.action_space)
        probs = [F.softmax(logit, dim = 0) for logit in logits]
        actions = [probs[0].argmax() for prob in probs]
        
        
        return [action.item() for action in actions]

    def calculate_loss(self, gamma=0.99):
        
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        device = self.state_values[0].get_device()
        rewards = torch.tensor(rewards, device = device)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value[0], reward)
            loss += (action_loss + value_loss)   
        
        return loss
    
    def clear_memory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

