# Qmix on defense-v1.py
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

# epsilon parameters class 
class epsilon_params():
    def __init__(self, A = 0.3, B = 0.1, C = 0.1):
        self.A = A
        self.B = B
        self.C = C

#################### class Q mix  ###########################################""
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
#################################### DQN Agent ####################################
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
#################################### Agent ########################################

class Agent():
    def __init__(self, name, net, **kwargs):
        self.name = name
        self.net = net
        self.target_net = copy.deepcopy(self.net)
        self.target_net.eval()
        self.epsilon_start  = kwargs.get('epsilon',0.99)
        _, self.id = self.name.split('agent_')
        self.device = kwargs.get('device','cuda')
        self.epsilon = self.epsilon_start
        

    def set_epsilon(self, N_episodes, episode, parameters):
        A = parameters.A
        B = parameters.B
        C = parameters.C
        standarized_time = (episode - A * N_episodes)/(B * N_episodes)
        cosh = np.cosh(math.exp(-standarized_time))
        epsilon = 1.1 - (1 /cosh + (episode * C / N_episodes))

        self.epsilon = epsilon

    def get_action(self, env, state):
        # here wehole have the hole state ==> so we just need the state of the corresponding agent 
        # so we need to reshape the state before getting the one wich interest us 
        if random.random() < self.epsilon:
            action = random.choice(range(env.action_space(env.agents[0]).n))
        else: 
            with torch.no_grad():
                state_tensor = torch.tensor(state, device = self.device)
                action = self.net(state_tensor).cpu().squeeze().argmax().numpy()

        return action


    def sync(self):
        self.target_net.load_state_dict(self.net.state_dict())


#################################### mixer  #####################################""  
class Mixer():
    def __init__(self, env, device = 'cpu', **kwargs):
        self.agents_name = env.agents
        self.n_agents = len(self.agents_name)
        self.n_team = int(self.n_agents / 2)

        self.action_space = env.action_space(self.agents_name[0]).n
        self.agent_state_shape = env.observation_space(self.agents_name[0])['obs'].shape[0]
        self.hidden_layer_dim = kwargs.get('hidden_layer_dim', 64)

        self.gamma = kwargs.get('gamma',0.9)
        self.learning_rate = kwargs.get('learning_rate',1e-4)

        self.device = device

        self.epsilon_parameters = kwargs.get('epsilon_parameters',epsilon_params(A = 0.3, B = 0.1, C = 0.1 ))
        self.dropout = kwargs.get('dropout',0.25)
        #DQN parameters 

        self.agent_net = DQN(input_nodes = self.agent_state_shape, hidden_nodes = 32, output_nodes = self.action_space, dropout = self.dropout).to(self.device)
        # the mixer will be used just for half of the team so we will need two nets and two target nets 
        self.net = QMixer(n_agents = self.n_team, hidden_layer_dim = self.hidden_layer_dim, state_shape = self.n_team * self.agent_state_shape,  dropout = self.dropout).to(self.device)
        self.target_net = copy.deepcopy(self.net)
        self.target_net.eval()

        # create the different agents | the training mode will be taken in the learn function 
        self.agents = {}
        for agent in self.agents_name:
            team, _ = agent.split('_')
            self.agents[agent] = Agent( name = agent, team = team, net = self.agent_net, device = self.device)

    
    # set which team gonna be training 

    
    def get_qvalues(self, batch, team):
        state = batch[0]
        len_s = state.size
        q = []
        for agent in self.agents_name:
            if self.agents[agent].team == team:
                s = [torch.tensor(state[idx][agent], device = self.device, dtype = torch.float32).unsqueeze(0) for idx in range(len_s)]
                s = torch.cat(s)
                q.append(self.agents[agent].net(s).unsqueeze(1))
        q_values = torch.cat(q, dim = 1)
        return q_values
            

    def get_target_q_values(self, batch, team):
        new_state = batch[4]
        len_s_ = new_state.size
        q = []
        for agent in self.agents_name:
            if self.agents[agent].team == team:
                s_ = [torch.tensor(new_state[idx][agent], device = self.device, dtype = torch.float32).unsqueeze(0) for idx in range(len_s_)]
                s_ = torch.cat(s_)
                q.append(self.agents[agent].target_net(s_).unsqueeze(1))
        q_target_values = torch.cat(q, dim = 1)
        return q_target_values        

    def set_epsilon(self, team, N_episodes, episode):
        for agent in self.agents:
            if agent.team == team:
                agent.set_epsilon(N_episodes, episode, self.epsilon_parameters)

    def sync(self, team):
        self.target_net.load_state_dict(self.net.state_dict()) 
        for agent in self.agents:
            if agent.team == team:
                agent.sync()

    # the batch is just created for the team that is in trainnig mode 
    def learn(self, team, batch):
        ## all the information we need 
        s = batch[0]
        size_s = s.size
        
        s = [torch.tensor(self.concat(s[idx]), device = self.device, dtype = torch.float32).unsqueeze(0) for idx in range(size_s)]
        s = torch.cat(s)
        # need the new state to compute the target 
        s_ = batch[4]
        size_s_ = s_.size
        s_ = [torch.tensor(self.concat(s_[idx]), device = self.device, dtype = torch.float32).unsqueeze(0) for idx in range(size_s_)]
        s_ = torch.cat(s_)

        # get the qvalues 
        Qvalues = self.get_qvalues(batch, team)
        actions = batch[2]
        actions = torch.cat([torch.tensor(np.array([actions[idx][agent] for agent in self.agent_names]), device = self.device, dtype = torch.float32).unsqueeze(0) for idx in range(actions.size)])
        actions  = actions.unsqueeze(2)
        q_values = Qvalues.gather(2,actions).squeeze(2).unsqueeze(1)
        Qtot = self.net(s, q_values).squeeze()
        # get the reward 
        reward = batch[3]
        reward = torch.tensor(np.array([reward[idx][list(reward.keys())[0]] for idx in range(reward.size)]), dtype= torch.float32, device = self.device)

        isdone = batch[5]
        isdone =  [isdone[idx][list(isdone.keys())[0]] for idx in range(isdone.size)]
        isdone = torch.tensor(isdone, device = self.device, dtype = torch.float32)

        Qtot[isdone] = 0

        with torch.no_grad():
            # get the q target 
            q_target_agents = torch.max(self.get_target_qvalues(batch), dim = 2)[0].unsqueeze(1)
            qtot_target  = self.target_net(s_, q_target_agents).squeeze()
            target  = reward + self.gamma * qtot_target.detach()
        
        loss = self.MSE(target, Qtot)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return actions

    def concat(self, dict):
        keys = dict.keys()
        elements = [dict[key] for key in keys]
        elements = np.concatenate(elements)
        return elements 

    

    
################################### Agent  #########################################
class Agent():
    def __init__(self, name, team, net, **kwargs): # kwargs : epsilon / device / 
        self.name = name
        self.team = team
        self.net = net
        self.target_net = copy.deepcopy(self.net)
        self.device = kwargs.get('device','cuda')
        self.epsilon = 1
    
    def set_epsilon(self, N_episodes, episode, parameters):
        A = parameters.A
        B = parameters.B
        C = parameters.C
        standarized_time = (episode - A * N_episodes)/(B * N_episodes)
        cosh = np.cosh(math.exp(-standarized_time))
        epsilon = 1.1 - (1 /cosh + (episode * C / N_episodes))
        
        self.epsilon = epsilon

    def random_action(self, state, is_done):
        mask = state['action_mask']
        p = mask / np.sum(mask)
        action = int(np.random.choice(range(len(p)), p = p)) if not is_done else None
        return action

    def get_action(self, team, state, is_done):
        # the state here is composed from two different information ( observation , action mask)
        # s = state['obs']
        s = state
        mask = state['action_mask']
        if self.team == team:
            epsilon = self.epsilon 
            if random.random() < epsilon:
                action = self.random_action(state, is_done)
            else:
                valid_action = False
                with torch.no_grad():
                    while not valid_action:
                        state_tensor = torch.tensor(s, device = self.device, dtype = torch.float32)
                        action = self.net(state_tensor).cpu().squeeze().argmax().numpy() if not is_done else None
                        if mask[action] == 1:
                            valid_action = True
                        else:
                            action = self.random_action(state, is_done)
                            valid_action = True
        else: 
            action = self.random_action(state, is_done)

    def sync(self):
        self.target_net.load_state_dict(self.net.state_dict())            


################################### Buffer / Experience ####################################
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

    
        




if __name__ == '__main__':
    # set experience 
    Experience = collections.namedtuple('Experience',['state','action_mask','action','reward','new_state','is_done'])
    # set the buffer 
    buffer = Buffer(capacity= 1000)
    env = defense_v0.env(terrain='central_7x7_2v2')
    env.reset()
    blue_team = ['blue_0', 'blue_1']
    red_team = ['red_0', 'red_1']
    # get the last agent 
    last_agent = env.agents[-1]
    agents_experience = {}
    creation = 0
    # we will create a dictionnary for each information we want to store 
    state = {}
    action_mask = {}
    action = {}
    reward = {}
    new_state = {}
    is_done = {}
    creation = 0
    # ####
    for agent in env.agent_iter():
        obs, r, done, _ = env.last()
        mask = obs['action_mask']
        last_agent = env.agents[-1]
        blue_team = ['blue_0','blue_1']
        red_team = ['red_0','red_1']

        if creation != 0:
            reward[agent] = r
            new_state[agent] = obs['obs']
            is_done[agent] = done
            experience = Experience(state, action_mask, action, reward, new_state, is_done)
            buffer.add(experience)

        state[agent] = obs['obs']
        action_mask[agent] = obs['action_mask']
        
        p = mask / np.sum(mask)
        act = int(np.random.choice(range(len(p)), p = p)) if not done else None
        action[agent] = act
        env.step(act)
        
        if agent == last_agent:
            creation += 1
    env.reset()
    batch = buffer.sample(10)
    mixer = Mixer(env, device = 'cpu' )
    a = 1
    