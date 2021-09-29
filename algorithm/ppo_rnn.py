from numpy.core.fromnumeric import clip
import torch
from torch._C import dtype
import torch.nn.functional as F
from torch import nn
from torch import distributions
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from time import perf_counter
from datetime import timedelta
class PPO(nn.Module):
    def __init__(self, observation_space, action_space, actor_lr, value_lr, batch_size, buffer_size, device):
        super(PPO, self).__init__()
        lookback = int((observation_space - 3) / 2)
        observation_space = {'price': lookback, 'volume': lookback, 'agent_state': 3}
        self.observation_space = observation_space
        self.action_space = action_space
        total_action_size = sum(action_space)
        # self.pv_affine = nn.GRU(input_size = 2, hidden_size = 32, num_layers = 1, batch_first = True)
        # self.all_affine = nn.Linear(32 + observation_space['agent_state'], 16)
        self.action_layer = nn.Sequential(
                              nn.Linear(32 + observation_space['agent_state'], 16),
                              nn.Tanh(),
                              nn.Linear(16, total_action_size),
                          )
        
        self.value_layer = nn.Sequential(
                            nn.Linear(32 + observation_space['agent_state'], 16),
                            nn.Tanh(),
                            nn.Linear(16, 1)
                        )

        self.action_module = nn.ModuleDict({
                              'pv': nn.GRU(input_size = 2, hidden_size = 32, num_layers = 1, batch_first = True),
                              'action': self.action_layer
                            })

        self.value_module = nn.ModuleDict({
                              'pv': nn.GRU(input_size = 2, hidden_size = 32, num_layers = 1, batch_first = True),
                              'value': self.value_layer
                            })                      

        self.actor_optimizer = Adam(self.action_layer.parameters(), actor_lr)
        self.value_optimizer = Adam(self.value_layer.parameters(), value_lr)
        self.activation_fn = torch.tanh
        self.buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.gamma = 0.99
        self.clip_range = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 1
        self.training_step = 0


    def get_action(self, state):
        lookback = self.observation_space['price']
        obs = torch.from_numpy(state).float().to(self.device)
        pv_obs = obs[:lookback*2].view(2, lookback)
        agent_state_obs = obs[-self.observation_space['agent_state']:]
        with torch.no_grad():
            output, _ = self.action_module['pv'](pv_obs.T.unsqueeze(0))
            pv_affine = torch.tanh(output.squeeze()[-1, :])
            all_affine = torch.cat([pv_affine, agent_state_obs])
            logits = self.action_module['action'](all_affine).split(self.action_space)
        
        probs = [F.softmax(logit, dim = 0) for logit in logits]
        dists = [Categorical(prob) for prob in probs]
        actions = [dist.sample().item() for dist in dists]
        action_prob = sum([dist.probs[action] for dist, action in zip(dists, actions)])
        
        return actions, action_prob.cpu().item()

    def get_action_probs(self, actions, states):
        lookback = self.observation_space['price']
        pv_obs = states[:, :lookback*2].view(-1, 2, lookback).transpose(2, 1)
        agent_state_obs = states[:, -self.observation_space['agent_state']:]
        output, _ = self.action_module['pv'](pv_obs)
        pv_affine = torch.tanh(output[:, -1, :])
        all_affine = torch.cat([pv_affine, agent_state_obs], dim = 1)
        logits = self.action_module['action'](all_affine).split(self.action_space, dim = 1)
        
        probs = [F.softmax(logit, dim = 1) for logit in logits]
        new_action_probs = sum([prob.gather(1, batch_actions_sep.view(-1, 1).long()) for prob, batch_actions_sep in zip(probs, actions.T)])

        return new_action_probs

    def get_value(self, states):
        lookback = self.observation_space['price']
        pv_obs = states[:, :lookback*2].view(-1, 2, lookback).transpose(2, 1)
        agent_state_obs = states[:, -self.observation_space['agent_state']:]
        output, _ = self.value_module['pv'](pv_obs)
        pv_affine = torch.tanh(output[:, -1, :])
        all_affine = torch.cat([pv_affine, agent_state_obs], dim = 1)
        values = self.value_module['value'](all_affine)
        return values

    def get_buffer_data(self):
        states, rewards, next_states, dones = [], [], [], []
        for transition in self.buffer:
            states.append(transition.state)
            rewards.append(transition.reward)
            next_states.append(transition.next_state)
            dones.append(transition.done)
        
        return [torch.tensor(item, dtype = torch.float, device = self.device) for item in [states, rewards, next_states, dones]]

    def get_mini_batch(self, batch_index):
        states, actions, log_probs = [], [], []
        for index in batch_index:
            transition = self.buffer[index]
            states.append(transition.state)
            actions.append(transition.action)
            log_probs.append(transition.log_prob)

        batch = [torch.tensor(item, dtype = torch.float, device = self.device) for item in [states, actions, log_probs]]
        return batch

    def update(self):
        # start_time = perf_counter()
        states, rewards, next_states, dones = self.get_buffer_data()
        values = self.get_value(states).detach()
        
        discounted_reward = 0
        returns = []

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1 and not dones[step]:
                discounted_reward = self.get_value(next_states[step].unsqueeze(0)).detach().item()
            if dones[step]:
                discounted_reward = 0
            discounted_reward = rewards[step] + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.stack(returns, dim = 0)
        

        for _ in range(10):
            for batch_index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                batch_states, batch_actions, old_log_probs = self.get_mini_batch(batch_index)
                batch_returns = returns[batch_index]
                old_values = values[batch_index]
                clip_range = self.clip_range
                new_values = self.get_value(batch_states)
                batch_advantages = (batch_returns - new_values).detach()
              
                # get new action probs
                new_action_probs = self.get_action_probs(batch_actions, batch_states)
                ratio = new_action_probs / old_log_probs.view(-1, 1)
                policy_loss_1 = batch_advantages * ratio
                policy_loss_2 = batch_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.action_module.parameters(), 0.5)
                self.actor_optimizer.step()
                # self.writer.add_scalar('loss/policy_loss', policy_loss, global_step=self.training_step)

                value_pred = torch.clamp(new_values, old_values - clip_range, old_values + clip_range)
                value_loss_1 = F.mse_loss(value_pred.squeeze(), batch_returns)
                value_loss_2 = F.mse_loss(new_values.squeeze(), batch_returns)
                value_loss = torch.max(value_loss_1, value_loss_2)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)

                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_module.parameters(), 0.5)
                self.value_optimizer.step()


                # total loss
                # loss = policy_loss + self.entropy_coef*entropy_loss + self.value_coef*value_loss 
                # loss = policy_loss + value_loss 
                # print(f"policy_loss: {round(policy_loss.item(), 5)}, entropy_loss: {round(self.entropy_coef*entropy_loss.item(), 2)}, value_loss: {round(self.value_coef*value_loss.item(), 2)}")
                # optimize
                # self.actor_optimizer.zero_grad()
                # self.value_optimizer.zero_grad()
                # loss.backward()
                # nn.utils.clip_grad_norm_(self.action_layer.parameters(), 0.5)
                # nn.utils.clip_grad_norm_(self.value_layer.parameters(), 0.5)
                # self.actor_optimizer.step()
                # self.value_optimizer.step()
                self.training_step += 1

        # cost_time = str(timedelta(seconds = perf_counter() - start_time))        
        # print(f"Run in {cost_time}.")
        # clear data
        del self.buffer[:]
        loss = {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item()}
        return loss


    
    def predict(self, state):
        obs = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            logits = self.action_layer(obs).split(self.action_space)
        
        probs = [F.softmax(logit, dim = 0) for logit in logits]
        actions = [prob.argmax().item() for prob in probs]

        return actions