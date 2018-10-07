import numpy as np
import brica

from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

# import copy


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


memory_size = 64
rollout_steps = 50
save_steps = 100
value_coeff = 0.5
entropy_coeff = 0.01
grad_norm_limit = 40
gamma = 0.99
lambd = 1.00
lr = 3e-4
input_channel = 6
task_num = 7


class ACModel(nn.Module):
    '''
    Actor-Critic Model
    '''
    def __init__(self, action_num=2, memory_size=64, d_limit=0.5):
        super().__init__()

        self.memory_size = memory_size
        self.action_num = action_num

        # Define image embedding
        self.image_embedding_size = 64
        self.concat_layer_size = 14400
        self.task_attention_linear = nn.Linear(7, input_channel)
        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channel+3, out_channels=16, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=self.image_embedding_size, kernel_size=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # Embedding Linear Layer
        self.embed_linear = nn.Linear(self.concat_layer_size+4096, self.image_embedding_size)
        
        # Define memory
        self.memory_rnn = nn.GRUCell(self.image_embedding_size, self.memory_size)

       
        # Resize image embedding
        self.embedding_size = self.memory_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_num),
        )
        self.actor_mu = nn.Sequential(
            nn.Linear(self.action_num, self.action_num),
            nn.Tanh()
        )
        self.actor_sigma = nn.Sequential(
            nn.Linear(self.action_num, self.action_num),
            nn.Softplus()
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def forward(self, obs, task, map, vgg_feature, memory):
        task_attention = self.task_attention_linear(task).view(-1, input_channel, 1, 1)
        obs = task_attention * obs
        obs = torch.cat([obs, map], dim=1)
        x = self.image_conv(obs)
        x = x.view(x.size()[0], -1)
        x = torch.cat([x, vgg_feature], dim=-1)
        x = self.embed_linear(x)
        hidden = self.memory_rnn(x, memory)
        embedding = hidden
        memory = hidden

        x = self.actor(embedding)
        mu = self.actor_mu(x)
        sigma = self.actor_sigma(x)
        dist_params = [mu, sigma]

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist_params, value, memory


class BG(object):
    def __init__(self, training=True, init_weight_path=None, use_cuda=False):
        self.timing = brica.Timing(5, 1, 0)
        self.training = training
        self.total_steps = 0
        self.ep_rewards = [0.]
        self.cuda = use_cuda
        self.ac_model = ACModel()
        if init_weight_path is not None:
            self.ac_model.load_state_dict(torch.load(init_weight_path))
        self.optimizer = optim.Adam(self.ac_model.parameters(), lr=lr)
        if self.cuda: self.ac_model = self.ac_model.cuda()
        self.init_params()

    def __call__(self, inputs):
        if 'from_environment' not in inputs:
            raise Exception('BG did not recieve from Environment')
        if 'from_pfc' not in inputs:
            raise Exception('BG did not recieve from PFC')
        if 'from_fef' not in inputs:
            raise Exception('BG did not recieve from FEF')
        

        vgg_feature, saliency_map, optical_flow, flow_salicency_map, saliency_gradient_map  = inputs['from_fef']
        task_phase = inputs['from_pfc'].detach()
        map_image = inputs['from_hp']
        map_image = np.array(map_image).reshape(128, 128, 3)
        saliency_map = saliency_map.reshape(128, 128, 1)
        optical_flow = optical_flow.reshape(128,128,2)
        flow_salicency_map = flow_salicency_map.reshape(128, 128, 1)
        saliency_gradient_map = saliency_gradient_map.reshape(128,128,2)
        concat_image = np.concatenate([
            saliency_map,
            optical_flow,
            flow_salicency_map,
            saliency_gradient_map,
        ], axis=-1)

        reward, done = inputs['from_environment']
        dones = [done]
        rewards = np.array([reward])

        # reset the LSTM state for done envs
        masks = (1. - torch.from_numpy(np.array(dones, dtype=np.float32).reshape(-1))).unsqueeze(1)
        if self.cuda: masks = masks.cuda()

        self.total_steps += 1
        self.ep_rewards = self.ep_rewards + rewards
        # if done:
        #     ep_rewards = 0
        rewards = torch.from_numpy(rewards).float().unsqueeze(1)
        if self.cuda: rewards = rewards.cuda()
        if self.prev_actions is not None:
            self.steps.append((
                rewards,
                masks,
                self.prev_actions.clone(),
                self.prev_policies,
                self.prev_values.clone()
            ))
        
        obs = np.expand_dims(concat_image, axis=0)
        obs = Variable(torch.from_numpy(obs.transpose((0, 3, 1, 2))).float())
        map_image = np.expand_dims(map_image, axis=0)
        map_image = Variable(torch.from_numpy(map_image.transpose((0, 3, 1, 2))).float())
        vgg_feature = Variable(vgg_feature.float())
        task_phase = Variable(task_phase.float())

        # network forward pasa
        dist_params, values, self.memory = self.ac_model(obs, task_phase, map_image, vgg_feature, self.memory)
        mu , sigma = dist_params
        policies = Normal(mu, sigma)
        actions = policies.sample()
        prob = policies.log_prob(actions)
        self.prev_actions = actions
        self.prev_policies = policies
        self.prev_values = values
        # print(self.total_steps)

        if self.total_steps % rollout_steps == 0 and self.training:
            self.update()
            self.init_params()
        
        if self.total_steps % save_steps == 0 and self.training:
            cur_weights = self.ac_model.state_dict()
            torch.save(cur_weights, "./data/bg.pth")

        
        return dict(to_sc=actions.cpu().numpy().reshape(-1, 1))
        

    def update(self):
        self.steps.append((None, None, None, None, self.prev_values.clone()))
        actions, values, returns, advantages, entropies = process_rollout(self.steps, self.cuda)
        # calculate action probabilities
        log_action_probs = self.prev_policies.log_prob(actions)

        policy_loss = (-log_action_probs * Variable(advantages)).sum()
        value_loss = (.5 * (values - Variable(returns)) ** 2.).sum()
        entropy_loss = (-1)*entropies.sum()
#         entropy_loss = (log_probs * probs).sum()

        loss = policy_loss + value_loss * value_coeff + entropy_loss * entropy_coeff
        loss.backward()

        nn.utils.clip_grad_norm(self.ac_model.parameters(), grad_norm_limit)
        self.optimizer.step()
        self.optimizer.zero_grad()
        print("total step", self.total_steps)
        print("Loss:", loss.data[0])
        print("Return:", torch.mean(returns).data[0])
        

    def init_params(self):
        self.steps = []
        self.memory = Variable(torch.zeros(1, memory_size))
        if self.cuda: self.memory = self.memory.cuda()
        self.prev_actions = None
        self.prev_policies = None
        self.prev_values = None


def process_rollout(steps, cuda, num_workers=1):
    # bootstrap discounted returns with final value estimates
    _, _, _, _, last_values = steps[-1]
    returns = last_values.data

    advantages = torch.zeros(num_workers, 1)
    if cuda: advantages = advantages.cuda()

    out = [None] * (len(steps) - 1)
    out_actions = [None] * (len(steps) - 1)
    out_policies = [None] * (len(steps) - 1)
    out_values = [None] * (len(steps) - 1)
    out_returns = [None] * (len(steps) - 1)
    out_advantages = [None] * (len(steps) - 1)
    out_entropies = [None] * (len(steps) - 1)

    # run Generalized Advantage Estimation, calculate returns, advantages
    for t in reversed(range(len(steps) - 1)):
        rewards, masks, actions, policies, values = steps[t]
        _, _, _, _, next_values = steps[t + 1]

        returns = rewards + returns * gamma * masks

        deltas = rewards + next_values.data * gamma * masks - values.data
        advantages = advantages * gamma * lambd * masks + deltas
        
        out_actions[t] = actions
        out_entropies[t] = policies.entropy()
        out_policies[t] = policies
        out_values[t] = values
        out_returns[t] = returns
        out_advantages[t] = advantages

    # return data as batched Tensors, Variables
    out_actions = torch.cat(out_actions, dim=0)
    out_values = torch.cat(out_values, dim=0)
    out_returns = torch.cat(out_returns, dim=0)
    out_advantages = torch.cat(out_advantages, dim=0)
    out_entropies = torch.cat(out_entropies, dim=0)
    return (out_actions, out_values, out_returns, out_advantages, out_entropies)
