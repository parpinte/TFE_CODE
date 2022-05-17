# python code for testing the different methods 

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
import pandas as pd
import statistics as st
import scipy.io
# to take screenshot
import pyautogui as auto
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
        if self.epsilon < 0:
            self.epsilon = 0.01
        if random.random() < self.epsilon:
            p = action_mask / np.sum(action_mask)
            action = int(np.random.choice(range(len(p)), p = p)) if not done else 0
            message = int(np.random.choice(range(len(p)), p = p)) if not done else 0
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
                message = Qm.cpu().squeeze().argmax().item() if not done else 0 # argmax
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
        self.INPUT_MESSAGES_BLUE = self.params['INPUT_MESSAGES_BLUE']
        self.INPUT_MESSAGES_RED = self.params['INPUT_MESSAGES_RED']
        self.OUTPUT_MESSAGES_RED = self.params['OUTPUT_MESSAGES_RED']
        self.OUTPUT_MESSAGES_BLUE = self.params['OUTPUT_MESSAGES_BLUE']
        self.HIDDEN_LAYER = self.params['HIDDEN_LAYER']
        # simulation 
        self.N_EPISODES = self.params['N_EPISODES']
        self.EPSILON_START = self.params['EPSILON_START']
        self.UPDATE_TIME = self.params['UPDATE_TIME']
        self.DEVICE_BLUE  = self.params['DEVICE_BLUE']
        self.DEVICE_RED  = self.params['DEVICE_RED']

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
        self.blue_net = DQN(input_nodes = self.STATE_SHAPE + self.INPUT_MESSAGES_BLUE, 
                hidden_nodes = self.HIDDEN_LAYER, output_nodes = self.ACTION_SPACE + self.OUTPUT_MESSAGES_BLUE, dropout = self.DROPOUT).to(self.DEVICE_BLUE)
        self.red_net = DQN(input_nodes = self.STATE_SHAPE + self.INPUT_MESSAGES_RED, 
                hidden_nodes = self.HIDDEN_LAYER, output_nodes = self.ACTION_SPACE + self.OUTPUT_MESSAGES_RED, dropout = self.DROPOUT).to(self.DEVICE_RED)
        self.agents = {}
        
        for agent in self.NAME_AGENTS:
            if 'blue' in agent:
                self.agents[agent] = Agent(agent, 
                                     net_parameters = [self.STATE_SHAPE, self.INPUT_MESSAGES_BLUE, self.HIDDEN_LAYER, self.ACTION_SPACE, self.OUTPUT_MESSAGES_BLUE], 
                                    net = self.blue_net, 
                                    epsilon_params = self.EPS, device = self.DEVICE_BLUE)
            if 'red' in agent:
                self.agents[agent] = Agent(agent, 
                                     net_parameters = [self.STATE_SHAPE, self.INPUT_MESSAGES_RED, self.HIDDEN_LAYER, self.ACTION_SPACE, self.OUTPUT_MESSAGES_RED], 
                                    net = self.red_net, 
                                    epsilon_params = self.EPS, device = self.DEVICE_RED)
           
        self.msg = {'blue' : 0, 'red': 0}

        
        
        
        
    
    def teams_creation(self):
        blue_team = []
        red_team = []
        for agent in self.env.agents:
            if 'blue' in agent:
                blue_team.append(agent)
            else: 
                red_team.append(agent)
        return blue_team, red_team

    def demo(self,N_times, blue_epsilon, red_epsilon, render):
        
        for agent in self.env.agents:
            if 'blue' in agent:
                self.agents[agent].epsilon = blue_epsilon
            if 'red' in agent:
                self.agents[agent].epsilon = red_epsilon

        cumulative_reward_blue  = []
        cumulative_reward_red = []
        
        messages_blue = []
        messages_red = []
        
        for _ in range(N_times): 
            self.env.reset()
            self.msg = {'blue' : 0, 'red': 0}
            message = 0
            n_cycles = []
            msg_red = []
            msg_blue = []
            cycle = 0
            r_blue = 0
            r_red = 0
            for agent in self.env.agent_iter():

                last_agent = self.env.agents[0]
                obs, r, done, _ = self.env.last()
                
                if 'blue' in agent:
                    r_blue += r
                    action, message = self.agents[agent].selector(obs, done, self.msg['blue'])
                    self.msg['blue'] = message
                    msg_blue.append((agent, message))
                    
                else:

                    r_red += r
                    action, message = self.agents[agent].selector(obs, done, self.msg['red'])
                    self.msg['red'] = message
                    msg_red.append((agent,message))

                self.env.step(action if not done else None)

                if render:
                    self.env.render()
            
                if agent == last_agent:
                    cycle += 1

            messages_blue.append(msg_blue)
            messages_red.append(msg_red)
            cumulative_reward_blue.append(r_blue/2)
            cumulative_reward_red.append(r_red/2)
            n_cycles.append(cycle)
        return messages_blue, messages_red, cumulative_reward_blue,cumulative_reward_red, n_cycles

class epsilon_params():
    def __init__(self, A = 0.3, B = 0.1, C = 0.1):
        self.A = A
        self.B = B
        self.C = C


def make_order(messages):
    blue_list = []
    red_list = []
    for idx in range(len(messages)):
        b_msg = []
        r_msg = []
        for index  in range(len(messages[idx])):
            if 'blue' in messages[idx][index][0] :
                b_msg.append(messages[idx][index][1])
            else:
                b_msg.append(messages[idx][index][1])
        blue_list.append(b_msg)
        red_list.append(r_msg)
    return blue_list, red_list

def variance(data):
     n = len(data)
     mean = sum(data) / n
     return sum((x - mean) ** 2 for x in data) / (n - 0)

def variance_computation(messages):
    var = []
    for idx, message in enumerate(messages):
        var.append(variance(message))
    return var

if __name__ == '__main__':
    configuration_file = r'configuration_launch_test.yaml'
    runner = Runner(configuration_file)
    # load the neural network parameters 
    file_path_blue = os.path.abspath('nets/RIAL/RIAL_10m_blue_up.pk')
    runner.blue_net.load_state_dict(torch.load(file_path_blue))
    file_path_red = os.path.abspath('nets/RIAL/ RIAL_2m_blue_down.pk')
    runner.red_net.load_state_dict(torch.load(file_path_red))
    
    N_times = 1
    blue_epsilon = 0
    red_epsilon = 0
    render = True
    messages_blue, messages_red, cumulative_reward_blue,cumulative_reward_red, n_cycles = runner.demo(N_times, blue_epsilon, red_epsilon, render)
    
    """
    a,b = make_order(messages_blue)
    # rearrenge the values to draw the messages 
    var_msg = np.array([st.pvariance(a[idx]) for idx in range(len(a))])
    cumulative_reward_blue = np.array(cumulative_reward_blue)
    win_blue = [cumulative_reward_blue[idx] > 0 for idx in range(len(cumulative_reward_blue))]
    fraction = sum(win_blue) / len(win_blue)
    print(fraction)
    n_cycles = np.array(n_cycles)
    scipy.io.savemat('RIAL-10m.mat', {'var_msg': var_msg,'cumulative_reward_blue' : cumulative_reward_blue ,'n_cycles' :n_cycles})
    """