# test of Q mix implementation 

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

import matplotlib.pyplot as plt



# Qmixer class 

class QMixer(nn.Module):
    def __init__(self, n_agents, hidden_layer_dim, state_shape):
        super(QMixer, self).__init__()
        # W1 have to be a matrix of nagents * hidden_layer_dim 
        # with pytorch we will create an output of n_agents * hidden_layer_dim and than we will do a reshape 
        self.n_agents = n_agents
        self.hidden_layer_dim = hidden_layer_dim
        self.state_shape = state_shape
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
        out = w2out.view(-1, 1, 1)
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
        state, action, reward,  new_state, is_done = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(new_state),
            np.array(is_done),
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

class VDN():
    def __init__(self):
        pass



# Agent
class Agent:
    def __init__(self, name, net, **kwargs):
        self.name = name
        self.net = net
        self.target_net = copy.deepcopy(self.net)
        self.target_net.eval()
        self.epsilon  = kwargs.get('epsilon',0.99)
        _, self.id = self.name.split('agent_')

    def set_epsilon(self, N_episodes, episode, end_rate):
        slope  = - self.epsilon / round(end_rate * N_episodes)
        const  = self.epsilon 
        if episode <= round(end_rate * N_episodes):
            eps = slope * episode + const
        else:
            eps = 0

        self.epsilon = eps

    def get_action(self, env, state):
        # here wehole have the hole state ==> so we just need the state of the corresponding agent 
        # so we need to reshape the state before getting the one wich interest us 
        if random.random() < self.epsilon:
            action = random.choice(range(env.action_space(env.agents[0]).n))
        else: 
            with torch.no_grad():
                action = self.net(state[self.name]).cpu().squeeze().argmax().numpy()

        return action


    def sync(self):
        self.target_net.load_state_dict(self.net.state_dict())

    



    








#  The Mixer 
class Mixer:
    def __init__(self, env, device= 'cpu', Mixer = 'Qmix', **kwargs):
        # need (agents , observation space , action space and state space in order to define all the nn ( agents + mixer )
        self.agent_names = copy.copy(env.agents)
        self.n_agents = len(self.agent_names)
        self.state_shape = env.observation_space(env.agents[0]).shape[0]
        self.action_space = env.action_space(env.agents[0]).n
        self.use_mixer = 'Qmix'
        # determine the different agents networkx
        self.agents = {}
        self.agent_net = DQN(input_nodes = self.state_shape, hidden_nodes = 64, output_nodes = self.action_space).to(device)
        # define the different agents 
        for agent in self.agent_names:
            self.agents[agent] = Agent(agent, self.agent_net, epsilon = 1)
        
        # ask which mixer we need to use 
        
        if self.mixer == 'Qmix':
            self.net = QMixer(n_agents = self.n_agents, hidden_layer_dim = 16, state_shape = self.n_agents * self.state_shape)
        elif self.mixer == 'VDN':
            self.net = VDN()

        # target net 
        self.target_net = copy.deepcopy(self.net)
        self.target_net.eval()

        # set the different parameters for the optimization 
        parameters = list(self.net.parameters())
        # add the parameters of the agent network 
        parameters += self.agent_net.parameters()

        # define the optimizer which will be used for the learning 
        self.lr = kwargs.get('lr', 0.001)
        self.gamma = kwargs.get('gamma',.9)

        self.optimizer = torch.optim.Adam(parameters, self.lr)

        # other usefaull parameters 
        self.epsilon = kwargs.get('epsilon',0.99)
        # loss function 
        self.MSE = nn.MSELoss()

    def sync(self):
        self.target_net.state_dict(self.net.load_state_dict) 
        # give order to sync for the different agents 
        for agent in self.agents.values():
            agent.sync()

    def epsilon_order(self,N_episodes, episode, end_rate):
        for agent in self.agents.values():
            agent.set_epsilon(N_episodes, episode, end_rate)


    def get_q_values(self, batch):
        # batch form = 'state','action','reward','new_state','is_done'
        # batch shape (BATCH_SIZE, dictionnary) 
        # we need just Qagents (BATCH_SIZE, n_agents)
        # we will first get only the observations which is the first dimension 
        s = batch[0] #  (batch size, dictionnary of the state of each agent )
        len_s = s.size
        #
        Q = []
        for agent in self.agent_names:
            s = [torch.tensor(s[idx][agent]).unsqueeze(0) for idx in range(len_s)]  # create the batch that will pass through the network  
            s = torch.cat(s)
            Q += self.agents[agent].net(s).unsqueeze(1)

        q_vals = torch.cat(Q, dim = 1)
        return q_vals

    def get_target_qvalues(self, batch):
        # the target q values will be the one to compute the target in order to compute the loss 
        s_ = batch[3]    # (batch size, dictionnary ) 
        len_s_ = s_.size

        Q_target = []
        for agent in self.agent_names:
            s_ = [torch.tensor(s_[idx][agent]).unsqueeze(0) for idx in range(len_s_)]  # create the batch that will pass through the network  
            s_ = torch.cat(s_)
            Q_target += self.agents[agent].target_net(s_).unsqueeze(1)

        q_target_vals = torch.cat(Q_target, dim = 1)
        return q_target_vals    


    def concat_batch(self, batch_s):
        pass 

    def learn(self, batch):
        # what we need for the learning function 
        # curret state that need to be concatenated for the qmix_net 
        s = batch[0] 
        # the state 
        s_ = batch[3]

        # get the qvalues 

        # get the q tot 

        # get the q target 

        # compute the loss 


        # back prop 

        





    

    
    





        
        



"""
    env = simple_spread_v2.parallel_env(N= 3, max_cycles=100)
    env.reset()
    print(env.action_space(env.agents[0]).n)

"""
class randomAgent():
    def __init__(self, actions):
        self.actions = actions 


    def get_action(self, observation):
        return random.choice(range(self.actions))





def concat(dict):
    keys = dict.keys()
    elements = [dict[key] for key in keys]
    elements = np.concatenate(elements)
    return elements  


if __name__ == '__main__':
    env = simple_spread_v2.parallel_env(N= 3, max_cycles=100)
    obs = env.reset()
    actions = env.action_space(env.agents[0]).n
    ag = randomAgent(actions)
    action = {}
    action  = {agent : ag.get_action(obs) for agent in env.agents}
    observation, reward, is_done, _ = env.step(action)
    # create an experience  
    Experience = collections.namedtuple('Experience',['state','action','reward','new_state','is_done'])
    exp = Experience(obs, action, reward, observation, is_done)
    # add experience to the buffer 
    buffer = Buffer(capacity= 10)
    for _ in range(100):
        buffer.add(exp)
 
    
    batch = buffer.sample(batch_size=10)
    b  = a














