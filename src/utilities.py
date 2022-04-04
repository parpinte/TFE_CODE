# Qmix on defense-v0.py
from re import I, S
import time
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections

import argparse
import copy
import math
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter 

import sys

from ray import get
sys.path.append('..')
from environment import defense_v0 
import numpy as np

# test of Q mix implementation on defense-v0 without communication 

from re import S
from pettingzoo.mpe import simple_spread_v2
import time
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections

import argparse
import copy
import math
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter 


# Qmixer class 

class QMixer(nn.Module):
    def __init__(self, n_agents, hidden_layer_dim, state_shape, **kwargs):
        super(QMixer, self).__init__()
        # W1 have to be a matrix of nagents * hidden_layer_dim 
        # with pytorch we will create an output of n_agents * hidden_layer_dim and than we will do a reshape 
        self.n_agents = n_agents
        self.hidden_layer_dim = hidden_layer_dim
        self.state_shape = state_shape
        self.ELU = nn.ELU()
        # create the hyper networks 
        self.w1_layer  = nn.Linear(self.state_shape, self.n_agents * self.hidden_layer_dim)
        self.w2_layer  = nn.Linear(self.state_shape, 1 * self.hidden_layer_dim) # size (batch , 1 )
        self.p = kwargs.get('dropout', 0.25)
        self.dropout = nn.Dropout(p = self.p)
        # consranare 
        self.b1 = nn.Linear(self.state_shape, self.hidden_layer_dim)
        # at b2 we have to get 1 output which is the Qtot 
        self.b2 = nn.Sequential(
                                nn.Dropout(p = self.p),
                                nn.Linear(self.state_shape, self.state_shape),   # , self.hidden_layer_dim
                                nn.ReLU(),
                                nn.Dropout(p = self.p),
                                nn.Linear(self.state_shape, 1)  # self.hidden_layer_dim, 1
                                )   


    # how do we have to do the forward ? 
    def forward(self, states, q_values):
        states = self.dropout(states)
        w1 = torch.abs(self.w1_layer(states))
        w1 = w1.reshape(-1, self.n_agents, self.hidden_layer_dim)

        w2 = torch.abs(self.w2_layer(states))
        w2 = w2.reshape(-1,self.hidden_layer_dim, 1)

        b1 = self.b1(states).reshape((-1, 1, self.hidden_layer_dim))

        b2 = self.b2(states).reshape((-1, 1, 1))

        out1 = self.ELU(torch.add(torch.bmm(q_values,w1), b1))
        Qtot = torch.add(torch.bmm(out1, w2), b2)
        Qtot = Qtot.reshape(-1)

        return Qtot


# Buffer 
class Buffer():
    # create the buffer size 
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen = capacity)
        
        
    # get buffer siz 
    def __len__(self):
        return len(self.buffer)
    # add an experience to the queue 

    def add(self,experience):
        self.buffer.append(experience)

    def show_content(self):
        for idx, exp in enumerate(self.buffer):
            print(f"Experience {idx} ==> {self.buffer[idx]}")
    # sample from the buffer 
    
    def sample(self, batch_size):
        # generate batch size indexes to get miltiple experiences 
        indices = np.random.choice(len(self.buffer), batch_size, replace=False) 
        # get the different varianbles 
        state, actions_mask, action, reward,  new_state, is_done = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(state),
            np.array(actions_mask),
            np.array(action),
            np.array(reward),
            np.array(new_state),
            np.array(is_done),
        )


# agent class ( DQN but it can be a GRU )

class DQN(nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, dropout):
        super(DQN, self).__init__()
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.dropout = nn.Dropout(p = dropout)
        self.linear = nn.Sequential(
            nn.Dropout(p = dropout),
            nn.Linear(self.input_nodes, self.hidden_nodes),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(self.hidden_nodes,self.hidden_nodes),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(self.hidden_nodes, self.output_nodes),
        )

    def forward(self, x):
        output = self.linear(x)
        return output


class VDN():
    def __init__(self):
        pass
    def forward(self):
        pass



# Agent

class Agent():
    def __init__(self, name, net, **kwargs):
        self.name = name
        self.net = net
        self.target_net = copy.deepcopy(self.net)
        self.target_net.eval()
        self.epsilon_start  = kwargs.get('epsilon',0.99)
        self.device = kwargs.get('device','cuda')
        self.epsilon = 1
        
    def reset_epsilon(self):
        self.epsilon = 1
    def set_epsilon(self, N_episodes, episode, parameters):
        A = parameters.A
        B = parameters.B
        C = parameters.C
        standarized_time = (episode - A * N_episodes)/(B * N_episodes)
        cosh = np.cosh(math.exp(-standarized_time))
        epsilon = 1.1 - (1 /cosh + (episode * C / N_episodes))

        self.epsilon = epsilon

    def sync(self):
        self.target_net.load_state_dict(self.net.state_dict())



    def get_action(self, state, done):
        s = state['obs']
        mask = state['action_mask']
        mask_tensor = torch.tensor(state['action_mask'], device = self.device, dtype = torch.bool)
        epsilon = self.epsilon
        if random.random() < epsilon:
            action = random_action(mask, done)
        else: 
            with torch.no_grad():

                state_tensor = torch.tensor(s, device = self.device, dtype = torch.float32)
                # state_tensor[not mask_tensor] = -1000
                q = self.net(state_tensor)
                q[~mask_tensor] = 0
                action = q.cpu().squeeze().argmax().numpy()
                


        return action


#  The Mixer 
class Mixer: # Mixer(env, team, device= 'cpu', mixer = 'Qmix', dropout)
    def __init__(self, env, team, device= 'cpu', mixer = 'Qmix', **kwargs):
        # need (agents , observation space , action space and state space in order to define all the nn ( agents + mixer )
        self.agent_names = copy.copy(team)
        self.n_agents = len(self.agent_names)
        self.state_shape = env.observation_space(self.agent_names[0])['obs'].shape[0]
        self.action_space = env.action_space(self.agent_names[0]).n
        self.use_mixer = mixer
        # self.epsilon = kwargs.get('epsilon',epsilon_params(A=0.3, B=0.1, C=0.1))
        self.device = device
        self.dropout = kwargs.get('dropout', 0.25)
        # determine the different agents networkx
        self.agents = {}
        self.agent_net = DQN(input_nodes = self.state_shape, hidden_nodes = 64, output_nodes = self.action_space, dropout = self.dropout).to(self.device)
        # define the different agents 
        for agent in self.agent_names:
            self.agents[agent] = Agent(agent, self.agent_net, device = self.device, dropout = self.dropout)
        
        #print(f'state_shape = {self.state_shape} | actions = {self.action_space}')
        
        # ask which mixer we need to use 
        
        if self.use_mixer == 'Qmix':
            self.net = QMixer(n_agents = self.n_agents, hidden_layer_dim = 32, state_shape = self.n_agents * self.state_shape, dropout = self.dropout).to(self.device)
        elif self.use_mixer == 'VDN':
            self.net = VDN()

        # target net 
        self.target_net = copy.deepcopy(self.net)
        self.target_net.eval()

        # set the different parameters for the optimization 
        parameters = list(self.net.parameters())
        # add the parameters of the agent network 
        parameters += self.agent_net.parameters()

        # define the optimizer which will be used for the learning 
        self.lr = kwargs.get('lr', 0.0001)
        self.gamma = kwargs.get('gamma',.9)

        self.optimizer = torch.optim.Adam(parameters, self.lr)

        # other usefaull parameters 
        self.epsilon = kwargs.get('epsilon',0.99)
        # loss function 
        self.MSE = nn.MSELoss()

    def sync(self):
        self.target_net.load_state_dict(self.net.state_dict()) 
        # give order to sync for the different agents 
        for agent in self.agents.values():
            agent.sync()

    def epsilon_order(self,N_episodes, episode, parameters_epsilon):
        for agent in self.agents.values():
            agent.set_epsilon(N_episodes, episode, parameters_epsilon)
        return self.agents[self.agents_name[0]].epsilon


    def get_q_values(self, batch):
        # batch form = 'state','action','reward','new_state','is_done'
        # batch shape (BATCH_SIZE, dictionnary) 
        # we need just Qagents (BATCH_SIZE, n_agents)
        # we will first get only the observations which is the first dimension 
        state = batch[0] #  (batch size, dictionnary of the state of each agent )
        len_s = state.size
        #
        Q = []
        for agent in self.agent_names:
            s = [torch.tensor(state[idx][agent], device = self.device, dtype = torch.float32).unsqueeze(0) for idx in range(len_s)]  # create the batch that will pass through the network  
            s = torch.cat(s)
            Q.append(self.agents[agent].net(s).unsqueeze(1))

        q_vals = torch.cat(Q, dim = 1)
        # print(q_vals)
        return q_vals

    def get_target_qvalues(self, batch):
        # the target q values will be the one to compute the target in order to compute the loss 
        state_ = batch[4]    # (batch size, dictionnary ) 
        len_s_ = state_.size
        # print(f'state target = {state_}')
        Q_target = []
        for agent in self.agent_names:
            s_ = [torch.tensor(state_[idx][agent], device = self.device, dtype = torch.float32).unsqueeze(0) for idx in range(len_s_)]  # create the batch that will pass through the network  
            s_ = torch.cat(s_) # len_s_ * 18
            Q_target.append( self.agents[agent].target_net(s_).unsqueeze(1))

        q_target_vals = torch.cat(Q_target, dim = 1)
        return q_target_vals  

    def update_agent(self):
        for agent in self.agent_names:
            self.agents[agent].net.load_state_dict(self.agent_net.state_dict()) 


    def concat(self, dict):
        keys = dict.keys()
        elements = [dict[key] for key in keys]
        elements = np.concatenate(elements)
        return elements  


    def learn(self, batch):
        # what we need for the learning function batch
        # curret state that need to be concatenated for the qmix_net 
        s = batch[0] 
        size_s = s.size
        s = [torch.tensor(self.concat(s[idx]), device = self.device, dtype = torch.float32).unsqueeze(0) for idx in range(size_s)]
        s = torch.cat(s) # ==> give tensor (batchSize, 54 ( 3 agents ))
        # print(f'state learn shape {s.shape}')

        # the state 
        s_ = batch[4]
        size_s_ = s_.size
        s_ = [torch.tensor(self.concat(s_[idx]), device = self.device, dtype= torch.float32).unsqueeze(0) for idx in range(size_s_)]
        s_ = torch.cat(s_) # ==> give tensor (batchSize, 54 ( 3 agents ))
        print(f'state_ learn shape {s_.shape}')

        # get the qvalues 
        Qvalues = self.get_q_values(batch)
        # print(f'Qvalues learn funct = {Qvalues.shape}')
        actions = batch[2]
        # print(f'actions example = {actions[1]}')
        actions = torch.cat([torch.tensor(np.array([actions[idx][agent] for agent in self.agent_names]), device = self.device, dtype = torch.int64).unsqueeze(0) for idx in range(actions.size)])
        # print(f'actions before qqueeze {actions.shape}')
        actions  = actions.unsqueeze(1)
        # print(f'actions shape {actions.shape}')
        q_values = Qvalues.gather(2,actions).squeeze(2).unsqueeze(1) # .squeeze(2) # need  to be verified 
        # get the q tot 
        # print(f'state shape {s.shape}')
        # print(s)
        Qtot = self.net(s, q_values).squeeze()
        print(f'Qtot = {Qtot}')

        # get the reward 

        reward = batch[3]
        reward = torch.tensor(np.array([reward[idx][self.agent_names[0]] for idx in range(reward.size)]), dtype= torch.float32, device = self.device)

        # is done needed 
        isdone = batch[5]
        isdone =  [isdone[idx][self.agent_names[0]] for idx in range(isdone.size)]
        isdone = torch.tensor(isdone, device = self.device)

        # update the Qtot 
        Qtot[isdone] = 0
        # no grad needed here so 
        with torch.no_grad():
            # get the q target 
            q_target_agents = torch.max(self.get_target_qvalues(batch), dim = 2)[0].unsqueeze(1)
            qtot_target  = self.target_net(s_, q_target_agents).squeeze()
            target  = reward + self.gamma * qtot_target.detach()
            # print(qtot_target)
        
        # compute the loss 
        loss = self.MSE(target, Qtot)
        # print(target)
        # loss = torch.tensor(loss.item())
        # loss.requires_grad = True
        # back prop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

######################### LIST OF FUNCTIONS ##################
def random_action(mask, is_done):
    p = mask / np.sum(mask)
    action = int(np.random.choice(range(len(p)), p = p)) if not is_done else None
    return action
################ EPSILON DECAY #########################
class epsilon_params():
    def __init__(self, A = 0.3, B = 0.1, C = 0.1):
        self.A = A
        self.B = B
        self.C = C



if __name__ == '__main__':
    pass