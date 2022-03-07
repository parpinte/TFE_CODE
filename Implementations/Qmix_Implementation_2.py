# test of Q mix implementation 

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

import matplotlib.pyplot as plt



# Qmixer class 

class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()
        # W1 have to be a matrix of nagents * hidden_layer_dim 
        # with pytorch we will create an output of n_agents * hidden_layer_dim and than we will do a reshape 
        self.n_agents = args.n_agents
        self.hidden_layer_dim = args.hidden_layer_dim
        self.state_shape = args.state_shape
        # create the hyper networks 
        self.w1_layer  = nn.Linear(self.state_shape, self.n_agents * self.hidden_layer_dim)
        self.w2_layer  = nn.Linear(self.state_shape, 1 * self.hidden_layer_dim) # size (batch , 1 )
        # consranare 
        self.b1 = nn.Linear(self.state_shape, self.hidden_layer_dim)
        # at b2 we have to get 1 output which is the Qtot 
        self.b2 = nn.Sequential(
                                nn.Linear(self.state_shape, self.hidden_layer_dim),
                                nn.ReLU(),
                                nn.Linear(self.hidden_layer_dim, 1) )   


    # how do we have to do the forward ? 
    def forward(self, states, q_values):
        w1 = torch.abs(self.w1_layer(states))
        
        w2 = torch.abs(self.w2_layer(states)).view(-1, self.hidden_layer_dim, 1 )
        b1 = self.b1(states).view(-1, 1, self.hidden_layer_dim)
        b2 = self.b2(states).view(-1, 1, 1)
        w1 = w1.view(-1, self.n_agents, self.hidden_layer_dim)
        w1b1_out = F.elu(torch.bmm(q_values, w1) + b1)  # F.elu(+ b1) 
        # apply w2
        w2out = torch.bmm(w1b1_out, w2) + b2
        prt = False 
        if prt:
            print(f"w1 output {w1.shape}")
            print(f"w2 output {w2.shape}")
            print(f"b1 reshape {b1.shape}")
            print(f"b2 reshape {b2.shape}")
            print(f"w1b1_out = {w1b1_out.shape}")
            print(f"w2out = {w2out.shape}")
        return w2out


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
        state, action, reward,  new_state, is_done, q_chosen = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(new_state),
            np.array(is_done),
            np.array(q_chosen)
        )


# agent class ( DQN but it can be a GRU )

class DQN(nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(DQN, self).__init__()
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.linear = nn.Sequential(
            nn.Linear(self.input_nodes, self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes,self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, self.output_nodes),
        )

    def forward(self, x):
        output = self.linear(x)
        return output
"""
une fois on a enregistré la derniere episode 

"""
# Agent
class agent:
    def __init__(self,name, net, **kwargs):
        self.name = name
        self.net = net
        self.target_net = copy.deepcopy(self.net)
        self.target_net.eval()








#  The Mixer 
class Mixer:
    def __init__(self,env , device= 'cpu', Mixer = 'Qmix', **kwargs):
        # need (agents , observation space , action space and state space in order to define all the nn ( agents + mixer )
        self.agent_names = copy.copy(env.agents)
        self.state_shape = env.observation_space(env.agents[0]).shape[0]
        self.action_space = env.action_space(env.agents[0]).n

        # determine the different agents 
        self.agents = {}
        agent_net = DQN(input_nodes = self.state_shape, hidden_nodes = 64, output_nodes = self.action_space).to(device)
        for agent in self.agent_names:
            self.agents[agent] = agent()




if __name__ == '__main__':
    env = simple_spread_v2.parallel_env(N= 3, max_cycles=100)
    env.reset()
    print(env.action_space(env.agents[0]).n)
















