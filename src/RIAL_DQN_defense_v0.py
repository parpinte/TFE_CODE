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


################### Agent ########################

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
        if self.epsilon < 0.05:
            self.epsilon = 0.05
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
                message = Qm.cpu().squeeze().argmax().item() if not done else 0
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
            action = Qa.cpu().squeeze().argmax().item() if not done else 0
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
        self.Big_buffer = Buffer(capacity = self.CAPACITY)
        self.remaining_opponent_agents = len(self.NAME_AGENTS)/2
        self.CAPACITY_AGENT = self.params['CAPACITY_AGENT']
        self.net_params = self.net.parameters()
        self.optimizer = torch.optim.Adam(self.net_params, self.LEARNING_RATE)
        for agent in self.NAME_AGENTS:
            self.agents[agent] = Agent(agent, 
                                     net_parameters = [self.STATE_SHAPE, self.INPUT_MESSAGES, self.HIDDEN_LAYER, self.ACTION_SPACE, self.OUTPUT_MESSAGES], 
                                    net = self.net, 
                                    epsilon_params = self.EPS, device = self.DEVICE)
            self.buffer[agent] = Buffer(capacity= self.CAPACITY_AGENT)
           

        self.Experience = collections.namedtuple('Experience',['state','action','reward','new_state','is_done','action_mask', 'message','next_message'])
        self.msg = {'blue' : 0, 'red': 0}
        self.MSE = nn.MSELoss()
        
        
        
        
    
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
        
        n_cycles = 0
        cum_reward = 0.0
        for agent in self.team_to_train(training_team):
            s[agent] , a[agent] ,re[agent], d[agent] ,am[agent] , m[agent] = [], [], [], [], [], []

        for agent in self.env.agent_iter():
            last_agent = self.env.agents[-1]
            # obs, r, done, _ = self.env.last()
            obs, r, done, _ = self.info(training_team)

            # print(self.env.agents)

            if training_team in agent:
                s[agent].append(obs['obs'])
                am[agent].append(obs['action_mask'] )
                re[agent].append(r)
                d[agent].append(done)
                self.agents[agent].set_epsilon(self.N_EPISODES, episode)
                # take the observation / the action and 
                m[agent].append(self.msg[training_team])
                action, message = self.agents[agent].selector(observation = obs, done = done, msg = self.msg[training_team])
                a[agent].append(message)
                self.msg[training_team] = message
                cum_reward += re[agent][-1]
            else: 
                self.agents[agent].epsilon = 1.0
                self.msg[self.other_team(training_team)] = 0
                action, message = self.agents[agent].selector(observation = obs, done = done, msg = 0) #self.msg[self.other_team(training_team)]
                message = 0 # so there is no communication done
                # fixe the opponent 
                # action = 0
                # 

            if agent == last_agent:
                n_cycles +=1
            
            self.env.step(action if not done else None)
            # self.env.render()
        # lsr = re['blue_0'][-1]
        # print(f'last reward {lsr}')
        # add the information to the batch 
        information_to_add = {}
        for agent in self.team_to_train(training_team):
            information_to_add[agent] = [s[agent],a[agent],re[agent]
                                    ,d[agent],am[agent],m[agent]]

        self.update_buffers(information_to_add, training_team)
        
        return cum_reward/2, n_cycles
            # stock avery transition 
    def other_team_members(self, training_team):
        
        if training_team == 'blue':
            return self.red_team
        else: 
            return self.blue_team   

    def remaining_enemy_agents(self, agents, training_team):
        en = [] 
        for agent   in agents:
            if not (training_team in agent):
                en.append(agent)
        return en

    def info(self, training_team):
        obs, r, done, i = self.env.last()
        remain_agents = self.env.agents
        enemy = self.remaining_enemy_agents(remain_agents, training_team)
        if len(enemy) >= 1 & len(enemy) < self.remaining_opponent_agents:
            r = 1
            self.remaining_opponent_agents = self.remaining_opponent_agents - 1

        return obs, r, done, i

        
    def update_buffers(self, information_to_add, training_team):
        for agent in self.team_to_train(training_team):
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
        team = self.team_to_train(training_team)
        for agent in team:
            self.agents[agent].sync()

    def train(self, training_team):
        writer = SummaryWriter('src/runs/Ep300000')
        for episode in range(self.N_EPISODES):
            r, n_cycles = self.generate(training_team, episode)
            self.mix_buffer(training_team)
            while self.BATCH_SIZE > self.Big_buffer.__len__():
                self.generate(training_team, episode)
                self.mix_buffer(training_team)

            batch = self.Big_buffer.sample(self.BATCH_SIZE)
            loss = self.learn(training_team, batch)
            if episode % (self.UPDATE_TIME) == 0:
                    self.sync(training_team)
                    
            if (episode % self.PRINT_TIME == 0) | (episode == self.N_EPISODES - 1):
                print(f'episode = {episode} | average reward {r} | loss = {loss} | epsilon = {self.agents[self.team_to_train(training_team)[0]].epsilon} | n_cycles = {n_cycles}')
                self.save(episode)

            writer.add_scalar('reward', r, episode)
            writer.add_scalar('loss', loss, episode)
            writer.add_scalar('epsilon',self.agents[self.team_to_train(training_team)[0]].epsilon, episode)
            writer.add_scalar('n_cycles', n_cycles, episode)
            

    def save(self, episode):
        filename = f"sim_indexComNoObstacles_{self.N_EPISODES}_{episode}.pk"
        torch.save(self.net.state_dict(), './nets/RIAL/ '+ filename)

    def learn(self, training_team, batch):
        # self.Experience(s[idx], a[idx], re[idx+1], s[idx+1], d[idx], am[idx],m[idx + 1])
        # current state
        state = batch[0]
        state_tensor = torch.tensor(state, device = self.DEVICE, dtype = torch.float32)
        # need the messages 
        messages = batch[6]
        message_tensor = torch.tensor(messages, device = self.DEVICE, dtype = torch.float32).unsqueeze(1)
        # next state
        state_ = batch[3]
        state_tensor_ = torch.tensor(state_, device = self.DEVICE, dtype = torch.float32)
        # get the q values 
        in_tensor = torch.cat((state_tensor, message_tensor),1)
        Qvalues = self.net(in_tensor)
        Qa = Qvalues[:,:self.ACTION_SPACE]
        # Qm = Qvalues[:,self.ACTION_SPACE:]
        # choose the Q values for the corresponding actions 
        # take the actions
        actions = batch[1]
        actions_tensor = torch.tensor(actions, device = self.DEVICE, dtype = torch.int64).unsqueeze(1)
        Qa = torch.gather(Qa, 1, actions_tensor).flatten()
        # get the reward 
        reward = batch[2] 
        # get target Q 
        is_done = batch[4]
        is_done_tensor = torch.tensor(is_done, device = self.DEVICE, dtype = torch.float32)
        indexes = torch.where(is_done_tensor == 0)[0]
        messages_next = batch[7]
        messages_next_tensor = torch.tensor(messages_next, device = self.DEVICE, dtype = torch.float32).unsqueeze(1)
        with torch.no_grad():
            reward_tensor = torch.tensor(reward, device = self.DEVICE, dtype = torch.float32)
            in_tensor_ = torch.cat((state_tensor_, messages_next_tensor),1)
            Qvalues_ = self.agents[self.team_to_train(training_team)[0]].target_net(in_tensor_)
            Qa_ = Qvalues_[:,:self.ACTION_SPACE]
            Q_target = Qa_.max(dim = 1)[0].flatten()
            reward_tensor = self.shape_reward(reward_tensor, indexes)
            target = reward_tensor + self.GAMMA * Q_target *  (1 - is_done_tensor)
            target = target.flatten()

        # compue the loss 
        criterion = nn.MSELoss()
        loss = criterion(target, Qa)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def shape_reward(self, tensor, idx):
        a = tensor[idx] < 0
        r = tensor[idx]
        r[a] = -1.0
        tensor[idx] = r

        return tensor

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
                    print(f"{agent} sent the message {action}")
                else:
                    # selector(self, observation, done, msg)
                    self.agents[agent].epsilon = 1
                    action, message = self.agents[agent].selector(obs, done, 0)
                    
                self.env.step(action if not done else None)
                self.env.render()
            inin = input('do you wonna run a demo ? ')
     

class epsilon_params():
    def __init__(self, A = 0.3, B = 0.1, C = 0.1):
        self.A = A
        self.B = B
        self.C = C


    

if __name__ == '__main__':
    configuration_file = r'configuration.yaml'
    runner = Runner(configuration_file)
    # runner.generate('blue', 1)
    # runner.demo(training_team = 'blue')
    
    eval = False
    if eval == True:
        file_path = os.path.abspath('nets/RIAL/sim_indexComNoObstacles_120000_119999.pk')
        runner.net.load_state_dict(torch.load(file_path))
        runner.demo(training_team = 'red')
    else:
        loss = runner.train('blue')
        runner.demo(training_team = 'blue')
   



