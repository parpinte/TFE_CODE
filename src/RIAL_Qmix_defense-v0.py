""" 
    Implementation of RIAL on defense-v0 with DQN. 
    we don't need the RNN because the environment is fully observable. 
    RIAL is based on IQL ( two different possiblities are offered for us)
    - IQL ( each agent has it's own network )
    - IQL but sharing parameters 
"""
from re import S
from ray import get


import os
import sys
sys.path.append('..')
import defense_v0 

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
import yaml

############################# Buffer class ####################################################"" 
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
        state, action, reward,  new_state, is_done, action_mask, message, next_message = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(new_state),
            np.array(is_done),
            np.array(action_mask),
            np.array(message),
            np.array(next_message)
            
        )
    def clear(self):
        self.buffer.clear()
################## buffer Qmix ####################
class Buffer_Qmix():
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
        state, action, reward,  new_state, is_done, action_mask, message, next_message, global_state ,global_state_next= zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(new_state),
            np.array(is_done),
            np.array(action_mask),
            np.array(message),
            np.array(next_message),
            np.array(global_state),
            np.array(global_state_next)
        )
    def clear(self):
        self.buffer.clear()


############### DQN Agent ####################
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
            nn.Linear(self.hidden_nodes,self.hidden_nodes),
            nn.ReLU(),
            nn.Dropout(p = dropout),
            nn.Linear(self.hidden_nodes, self.output_nodes),
        )

    def forward(self, x):
        output = self.linear(x)
        return output

############################# Qmix ######################################""
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

#####

class Agent():
    def __init__(self, name, net_parameters, net, epsilon_params,**kwargs):
        self.name = name
        # net parameters 
        self.input_state = net_parameters[0]
        self.input_message = net_parameters[1]
        self.hidden_state = net_parameters[2]
        self.n_actions = net_parameters[3]
        self.n_messages = net_parameters[4]
        self.dropout = kwargs.get('dropout', 0.2)
        # 
        self.net = net
        self.target_net = copy.deepcopy(self.net)
        self.target_net.eval()
        self.device = kwargs.pop('device', 'cpu')
        self.epsilon = 1.0
        self.eps_parameters = epsilon_params

    def sync(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def selector(self, observation, done, msg):
        s = observation['obs']
        action_mask = observation['action_mask']
        if self.epsilon < 0:
            self.epsilon = 0.01
        if random.random() < self.epsilon:
            p = action_mask / np.sum(action_mask)
            action = int(np.random.choice(range(len(p)), p = p)) if not done else 0
            message = random.choice(range(self.n_messages)) if not done else 0
        else: 
            with torch.no_grad():
                mask_tensor = torch.tensor(action_mask, dtype = torch.bool)
                state = s
                state = np.append(state, msg)
                state = torch.tensor(state, device = self.device, dtype = torch.float32)
                qma = self.net(state)
                # the Q values will be devided into Qa and Qm 
                Qa = qma[:self.n_actions]
                Qa[~ mask_tensor] = min(Qa).cpu().detach().item()
                Qm = qma[self.n_actions:]
                # from Qa we will choose the action and from Qm we will choose the message
                action = Qa.cpu().squeeze().argmax().item() if not done else 0
                message = Qm.cpu().squeeze().max().item() if not done else 0 # argmax
        return action, message
            
    def set_epsilon(self, N_episodes, episode):
        A = self.eps_parameters.A
        B = self.eps_parameters.B
        C = self.eps_parameters.C
        standarized_time = (episode - A * N_episodes)/(B * N_episodes)
        cosh = np.cosh(math.exp(-standarized_time))
        epsilon = 1.1 - (1 /cosh + (episode * C / N_episodes))

        self.epsilon = epsilon

    def get_target_action(self, observation, done, msg):
        action_mask = observation['action_mask']
        mask_tensor = torch.tensor(action_mask, dtype = torch.bool)
        state = observation['obs']

        state = np.append(state, msg)
        with torch.no_grad():
            state = torch.tensor(state, device = self.device, dtype = torch.float32)
            qma = self.target_net(state)
            # the Q values will be devided into Qa and Qm 
            Qa = qma[:self.n_actions]
            Qa[~ mask_tensor] = min(Qa).cpu().detach().item()
            Qm = qma[self.n_actions:]
            # from Qa we will choose the action and from Qm we will choose the message
            action = Qa.cpu().squeeze().argmax().item() if not done else None
            message = Qm.cpu().squeeze().argmax().item() if not done else 0
        return action, message

    
class Runner():
    def __init__(self, configuration_file ):
        with open(configuration_file) as file:
            self.params = yaml.load(file, Loader = yaml.FullLoader)
        self.TERRAIN = self.params['TERRAIN']
        self.MAX_CYCLES = self.params['MAX_CYCLES']
        self.INPUT_MESSAGES = self.params['INPUT_MESSAGES']
        self.OUTPUT_MESSAGES = self.params['OUTPUT_MESSAGES']
        self.HIDDEN_LAYER = self.params['HIDDEN_LAYER']
        # simulation 
        self.N_EPISODES = self.params['N_EPISODES']
        self.EPSILON_START = self.params['EPSILON_START']
        self.UPDATE_TIME = self.params['UPDATE_TIME']
        self.DEVICE  = self.params['DEVICE']

        # EPSILON
        self.EPS = epsilon_params(self.params['A'],self.params['B'],self.params['C'])

        # BUFFER PARAMETRS
        self.CAPACITY = self.params['CAPACITY']
        self.BATCH_SIZE = self.params['BATCH_SIZE']

        # NETWORKX
        self.LEARNING_RATE = self.params['LEARNING_RATE']
        self.GAMMA = self.params['GAMMA']
        self.DROPOUT = self.params['DROPOUT']
        self.PRINT_TIME = self.params['PRINT_TIME']
        self.env = defense_v0.env(terrain= self.TERRAIN, max_cycles = self.MAX_CYCLES)
        self.env.reset()
        self.blue_team, self.red_team = self.teams_creation()
        self.NAME_AGENTS = self.env.agents
        self.STATE_SHAPE = self.env.observation_space(self.NAME_AGENTS[0])['obs'].shape[0]
        self.ACTION_SPACE = self.env.action_space(self.NAME_AGENTS[0]).n
        self.net = DQN(input_nodes = self.STATE_SHAPE + self.INPUT_MESSAGES, hidden_nodes = self.HIDDEN_LAYER, output_nodes = self.ACTION_SPACE + self.OUTPUT_MESSAGES, dropout = self.DROPOUT).to(self.DEVICE)
        self.agents = {}
        self.buffer = {}
        self.Big_buffer = Buffer_Qmix(capacity = self.CAPACITY)
        self.remaining_opponent_agents = len(self.NAME_AGENTS)/2
        self.CAPACITY_AGENT = self.params['CAPACITY_AGENT']
        
        for agent in self.NAME_AGENTS:
            self.agents[agent] = Agent(agent, 
                                     net_parameters = [self.STATE_SHAPE, self.INPUT_MESSAGES, self.HIDDEN_LAYER, self.ACTION_SPACE, self.OUTPUT_MESSAGES], 
                                    net = self.net, 
                                    epsilon_params = self.EPS, device = self.DEVICE)
            self.buffer[agent] = Buffer(capacity= self.CAPACITY_AGENT)
           
        self.Qmix_experience = collections.namedtuple('Experience',['state','action','reward','new_state',
                'is_done','action_mask', 'message','next_message', 'total_state','global_state_next'])
        self.Experience = collections.namedtuple('Experience',['state','action','reward','new_state',
                            'is_done','action_mask', 'message','next_message'])
        self.msg = {'blue' : 0, 'red': 0}
        self.MSE = nn.MSELoss()
        self.mixer = QMixer(n_agents = int(len(self.NAME_AGENTS)/2), hidden_layer_dim = self.HIDDEN_LAYER, state_shape = self.STATE_SHAPE, dropout = self.DROPOUT).to(self.DEVICE)
        self.mixer_target = copy.deepcopy(self.mixer)
        self.mixer_target.eval()
        self.net_params = list(self.net.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(self.net_params, self.LEARNING_RATE)
        
        
        
    
    def teams_creation(self):
        blue_team = []
        red_team = []
        for agent in self.env.agents:
            if 'blue' in agent:
                blue_team.append(agent)
            else: 
                red_team.append(agent)
        return blue_team, red_team

    def generate(self, training_team, episode): # take into account the communication 
        self.env.reset()
        n = int(len(self.env.agents))
        s,a,re,d,am,m = {},{},{},{},{},{}
        global_state, global_state_next = [],[]
        n_cycles = 0
        cum_reward = 0.0
        for agent in self.NAME_AGENTS:
            s[agent] , a[agent] ,re[agent], d[agent] ,am[agent] , m[agent] = [], [], [], [], [], []


        for agent in self.env.agent_iter():
            last_agent = self.env.agents[-1]
            # obs, r, done, _ = self.env.last()

            obs, r, done, _ = self.info(training_team)
            s[agent].append(obs['obs'])
            am[agent].append(obs['action_mask'] )
            re[agent].append(r)
            d[agent].append(done)
            if training_team in agent:
                self.agents[agent].set_epsilon(self.N_EPISODES, episode)
                # take the observation / the action and 
                m[agent].append(self.msg[training_team])
                action, message = self.agents[agent].selector(observation = obs, done = done, msg = self.msg[training_team])
                a[agent].append(action)
                self.msg[training_team] = message
                cum_reward += re[agent][-1]
            else: 
                self.agents[agent].epsilon = 1
                self.msg[self.other_team(training_team)] = 0
                action, message = self.agents[agent].selector(observation = obs, done = done, msg = self.msg[self.other_team(training_team)])
                message = 0 # so there is no communication done
                # fixe the opponent 
                # action = 0
                # 
                a[agent].append(action) 
                m[agent].append(message)
            # print(self.survivors(training_team))
            if agent in self.survivors(training_team):
                if (agent == self.survivors(training_team)[0]):
                    global_state.append(self.env.state())

            

            if agent == last_agent:
                n_cycles +=1

            
                
            
            self.env.step(action if not done else None)
        # lsr = re['blue_0'][-1]
        # print(f'last reward {lsr}')
        # add the information to the batch 
        information_to_add = {}
        for agent in self.NAME_AGENTS:
            information_to_add[agent] = [s[agent],a[agent],re[agent]
                                    ,d[agent],am[agent],m[agent]]

        self.update_buffers(information_to_add)
        
        return cum_reward/2, n_cycles, global_state
            # stock avery transition 
    def other_team_members(self, training_team):
        
        if training_team == 'blue':
            return self.red_team
        else: 
            return self.blue_team   

    def info(self, training_team):
        obs, r, done, i = self.env.last()
        enemy = self.other_team_members(training_team)
        if (len(enemy) > 1) & (len(enemy) < (self.remaining_opponent_agents)):
            r = 0.3
            self.remaining_opponent_agents = self.remaining_opponent_agents - 1
        return obs, r, done, i

        
    def update_buffers(self, information_to_add):
        for agent in self.NAME_AGENTS:
            s = information_to_add[agent][0]
            a = information_to_add[agent][1]
            re = information_to_add[agent][2]
            d = information_to_add[agent][3]
            am = information_to_add[agent][4]
            m = information_to_add[agent][5]
            
            for idx in range(len(s)-1):
                experience = self.Experience(s[idx], a[idx], re[idx+1], s[idx+1], d[idx + 1], am[idx], m[idx], m[idx + 1])
                self.buffer[agent].add(experience)

    def mix_buffer(self, training_team):
        team = self.team_to_train(training_team)
        for agent in team:
            for idx in range(self.buffer[agent].__len__()):
                self.Big_buffer.add(self.buffer[agent].buffer[idx])
            self.buffer[agent].clear()

    def other_team(self, training_team):
        if training_team == 'blue':
            return 'red'   
        else: 
            return 'blue'

    def team_to_train(self, team):
        if team == 'blue':
            return self.blue_team
        else: 
            return self.red_team
    def sync(self, training_team):
        self.mixer_target.load_state_dict(self.mixer.state_dict())
        team = self.team_to_train(training_team)
        for agent in team:
            self.agents[agent].sync()

    
            

    def save(self, episode):
        filename = f"sim_Qmixcenral_7x7.pk"
        torch.save(self.net.state_dict(), './nets/RIAL/ '+ filename)


    def reset_epsilon(self, training_team):
        for agent in self.NAME_AGENTS:
            if training_team in agent:
                self.agents[agent].epsilon = 0.0
            else:
                self.agents[agent].epsilon = 1.0

    def demo(self,training_team):
        inin = input('do you wonna run a demo ? ')
        while inin == 'y':
            self.reset_epsilon(training_team)
            self.env.reset()
            self.msg = {'blue' : 0, 'red': 0}
            message = 0
            for agent in self.env.agent_iter():
                obs, _, done, _ = self.env.last()
                if training_team in agent:
                    action, message = self.agents[agent].get_target_action(obs, done, self.msg[training_team])
                    self.msg[training_team] = message
                else:
                    # selector(self, observation, done, msg)
                    action, message = self.agents[agent].selector(obs, done, self.msg[self.other_team(training_team)])
                    self.msg[self.other_team(training_team)] = 0
                self.env.step(action if not done else None)
                self.env.render()
            inin = input('do you wonna run a demo ? ')
    def team_buffers(self, training_team):
        self.team_buffer = {}
        for agent in self.NAME_AGENTS:
            if training_team in agent:
                self.team_buffer[agent] = copy.deepcopy(self.buffer[agent])
    def min_length_buffer(self):
        pass
    def mix_buffers(self):
        self.team_buffer = self.team_buffer
        min_len = min(self.team.buffer.values)
        pass

    

    def MixerQmix(self, training_team, global_state):
        
        # get the training team
        team = self.team_to_train(training_team)
        
        max_size = 0
        # get the minimum buffer size 
        for agent in team:
            if self.buffer[agent].__len__() > max_size:
                max_size = self.buffer[agent].__len__()
            
        
        for idx in range(max_size):
            state, action, reward, new_state,is_done,action_mask,message,next_message = [],[], [], [], [], [], [], []
            for agent in team:
                if idx <= self.buffer[agent].__len__() - 1:
                    state.append(self.buffer[agent].buffer[idx].state)
                    action.append(self.buffer[agent].buffer[idx].action)
                    new_state.append(self.buffer[agent].buffer[idx].new_state)
                    reward.append(self.buffer[agent].buffer[idx].reward)
                    is_done.append(self.buffer[agent].buffer[idx].is_done)
                    action_mask.append(self.buffer[agent].buffer[idx].action_mask)
                    message.append(self.buffer[agent].buffer[idx].message)
                    next_message.append(self.buffer[agent].buffer[idx].next_message)
                else:
                    state.append(self.buffer[agent].buffer[-1].state) #np.zeros(self.STATE_SHAPE))
                    action.append(0)
                    new_state.append(self.buffer[agent].buffer[-1].state)#np.zeros(self.STATE_SHAPE))
                    reward.append(0)
                    is_done.append(1)
                    action_mask.append(np.ones(self.ACTION_SPACE))
                    message.append(0)
                    next_message.append(0)

            state = np.array(state)
            action_mask = np.array(action_mask)
            action = np.array(action)
            new_state = np.array(new_state)
            reward = np.array(reward)
            message = np.array(message)
            next_message = np.array(next_message)
            is_done = np.array(is_done)
            self.Big_buffer.add(self.Qmix_experience(state,action,reward,new_state,
                        is_done,action,message,next_message, global_state[idx] ,global_state[idx+ 1]))

        for agent in team:
            self.buffer[agent].clear()

    def train_from_Qmix(self, training_team):
        writer = SummaryWriter('src/runs/central_10x10_Qmix')
        for episode in range(self.N_EPISODES):
            r, n_cycles, global_state = self.generate(training_team, episode)
            self.MixerQmix(training_team,global_state)
            while self.BATCH_SIZE > self.Big_buffer.__len__():
                _,_, global_state = self.generate(training_team, episode)
                self.MixerQmix(training_team, global_state)
            
            batch = self.Big_buffer.sample(self.BATCH_SIZE)
            loss = self.learn_from_Qmix(training_team, batch)
            if episode % (self.UPDATE_TIME) == 0:
                    self.sync(training_team)
                    
            if (episode % self.PRINT_TIME == 0) | (episode == self.N_EPISODES - 1):
                print(f'episode = {episode} | average reward {r} | loss = {loss} | epsilon = {self.agents[self.team_to_train(training_team)[0]].epsilon} | n_cycles = {n_cycles}')
                self.save(episode)

            writer.add_scalar('reward', r, episode)
            writer.add_scalar('loss', loss, episode)
            writer.add_scalar('epsilon',self.agents[self.team_to_train(training_team)[0]].epsilon, episode)
            writer.add_scalar('n_cycles', n_cycles, episode)

    def survivors(self, training_team):
        remaining_agents = self.env.agents
        s = []
        for agent in remaining_agents:
            if training_team in agent:
                s.append(agent)
        return s

    def learn_from_Qmix(self, training_team, batch):
        state = batch[0]
        state_tensor = torch.tensor(state, device = self.DEVICE, dtype = torch.float32) # (batch, 2 agent , size obs)
        messages = batch[6]
        messages_tensor = torch.tensor(messages, device = self.DEVICE, dtype = torch.float32).unsqueeze(2) #(batch, n agents,1)
        state_ = batch[3]
        state_tensor_ = torch.tensor(state_, device = self.DEVICE, dtype = torch.float32) # (batch, n agents, size obs)
        in_tensor = torch.cat((state_tensor, messages_tensor),2)
        Qvalues = self.net(in_tensor)
        Qa = Qvalues[:,:,:self.ACTION_SPACE]
        # take the different actions
        actions = batch[1]
        actions_tensor = torch.tensor(actions, device = self.DEVICE, dtype= torch.int64).unsqueeze(2)
        Qa = torch.gather(Qa, 2, actions_tensor).squeeze(2)
        global_state = batch[8]
        global_state_tensor = torch.tensor(global_state, device = self.DEVICE, dtype= torch.float32)
        Qa = Qa.unsqueeze(1)
        Qmix = self.mixer(global_state_tensor, Qa)
        reward = batch[2] 
        is_done = batch[4]
        is_done_tensor = torch.tensor(is_done, device = self.DEVICE, dtype = torch.float32)
        messages_next = batch[7]
        messages_next_tensor = torch.tensor(messages_next, device = self.DEVICE, dtype = torch.float32).unsqueeze(2)
        global_state_next = batch[9]
        global_state_next_tensor = torch.tensor(global_state_next, device =self.DEVICE, dtype = torch.float32)
        with torch.no_grad():
            is_done_tensor = torch.tensor(is_done_tensor, device = self.DEVICE, dtype = torch.bool)
            reward_tensor = torch.tensor(reward, device = self.DEVICE, dtype = torch.float32)
            in_tensor_ = torch.cat((state_tensor_, messages_next_tensor),2)
            Qvalues_ = self.agents[self.team_to_train(training_team)[0]].target_net(in_tensor_)
            Qa_ = (Qvalues_[:,:,:self.ACTION_SPACE])
            Q_target = Qa_.max(dim = 2)[0]
            Q_target[is_done_tensor] = 0
            Q_target = Q_target.unsqueeze(1)
            Q_target_mixer = self.mixer_target(global_state_next_tensor, Q_target)
            reward_qmix = reward_tensor.mean(dim = 1)
            target = reward_qmix + self.GAMMA * Q_target_mixer
        
        criterian = nn.MSELoss()
        loss = criterian(Qmix, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


class epsilon_params():
    def __init__(self, A = 0.3, B = 0.1, C = 0.1):
        self.A = A
        self.B = B
        self.C = C


    

if __name__ == '__main__':
    configuration_file = r'configuration_Qmix.yaml'
    runner = Runner(configuration_file)
    # runner.generate('blue', 1)
    # runner.demo(training_team = 'blue')
    
    eval = False
    if eval == True:
        file_path = os.path.abspath('nets/RIAL/ sim_VDNcenral_7x7.pk')
        runner.net.load_state_dict(torch.load(file_path))
        runner.sync('blue')
        runner.demo(training_team = 'blue')

    else:
        runner.train_from_Qmix('blue')
        runner.demo(training_team = 'blue')
   

"""

state.append(self.buffer[agent].buffer[idx-1].state)
                    action.append(self.buffer[agent].buffer[idx-1].action)
                    new_state.append(self.buffer[agent].buffer[idx-1].new_state)
                    reward.append(self.buffer[agent].buffer[idx-1].reward)
                    is_done.append(self.buffer[agent].buffer[idx-1].is_done)
                    action_mask.append(self.buffer[agent].buffer[idx-1].action_mask)
                    message.append(self.buffer[agent].buffer[idx-1].message)
                    next_message.append(self.buffer[agent].buffer[idx-1].next_message)
"""

