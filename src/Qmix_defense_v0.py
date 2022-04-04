# qmix defense v0 
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

import utilities as u




def create_team(agents):
    blue_team = []
    red_team = []
    for agent in agents:
        
        if 'blue' in agent: 
            blue_team.append(agent)
        else:
            red_team.append(agent)
    return blue_team, red_team

def generate(env, mixer, team, buffer, episode, N_episodes, epsilon_parameters):
    env.reset()
    cum_reward = 0.0
    state = {}
    action_mask = {}
    action = {}
    reward = {}
    new_state = {}
    is_done = {}
    creation = 0
    last_agent = env.agents[-1]
    for agent in env.agent_iter():
        obs, r, done, _ = env.last()
        cum_reward += r
        # print(f'agent = {agent} done? => {done}')
        mask = obs['action_mask']
        # last_agent = env.agents[-1]
        agent_team,_ = agent.split('_')
        agent_team = str(agent_team)
        # print(agent_team)
        if creation != 0:
            if team in agent:
                reward[agent] = r
                new_state[agent] = obs['obs']
                is_done[agent] = done
                experience = Experience(copy.deepcopy(state), copy.deepcopy(action_mask), copy.deepcopy(action), copy.deepcopy(reward), copy.deepcopy(new_state), copy.deepcopy(is_done))
                # print(experience)
                buffer[team].add(experience) 
        if team in agent:
            state[agent] = obs['obs']
            action_mask[agent] = obs['action_mask'] 

        if team in agent:
            # if the agent has to train then 
            # set the epsilon value 
            mixer[agent_team].agents[agent].set_epsilon(N_episodes, episode, epsilon_parameters)
            eps = mixer[agent_team].agents[agent].epsilon
            # get an action 
            act = (mixer[agent_team].agents[agent].get_action(obs, done))
            print(f'{agent} took action : {act}')
            msk = obs['action_mask']
            # print(f'{agent} action mask = {msk}')
            print(f'type of the variable = {type(act)}')
            # apply the action 
            action[agent] = act
            env.step(act)
            
            
                

        else: 
            act = mixer[agent_team].agents[agent].get_action(obs, done)
            # print(f'{agent} took action : {act}')
            msk = obs['action_mask']
            # print(f'{agent} action mask = {msk}')
            # print(f'type of the action = {type(act)}')
            # print(f'action = {act}')
            # print(f'action for agent = {agent} ==> {act}')
            env.step(act)

        if agent == last_agent:
                creation += 1
        
        # env.render()
        # env.render()
    return eps, cum_reward
        






# parameters
N_EPISODES = 10

# buffer
CAPACITY = 200
BATCH_SIZE = 11


# network parameters 
DROPOUT = 0.25
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
GAMMA = 0.9

# terrain 
TERRAIN = 'central01_10x10'
MAX_CYCLES = 30


if __name__ == '__main__':
    env = defense_v0.env(terrain=TERRAIN, max_cycles = MAX_CYCLES)
    env.reset()
    agent_names = env.agents
    print(DEVICE)
    print(f'agent_names = {agent_names}')
    print('Terrain \n')
    print(TERRAIN + '.ter')
    blue_team, red_team = create_team(agent_names) 
    Experience = collections.namedtuple('Experience',['state','action_mask','action','reward','new_state','is_done'])
    team =  'blue'
    buffer = {'blue': u.Buffer(capacity = CAPACITY), 'red' : u.Buffer(capacity = CAPACITY)}
    mixer = {}
    mixer['blue'] = u.Mixer(env = env, team = blue_team, device= 'cpu', mixer = 'Qmix', dropout = DROPOUT, lr = LEARNING_RATE, gamma = GAMMA)
    mixer['red'] = u.Mixer(env = env, team = red_team, device= 'cpu', mixer = 'Qmix', dropout = DROPOUT,lr = LEARNING_RATE, gamma = GAMMA)
    # print(mixer['red'].agent_names)
    epsilon_parameters = u.epsilon_params(A=0.3, B=0.1, C=0.1)
    # team to train is
    
    batch = {}
    for episode in range(0, N_EPISODES):
        eps, cum_reward = generate(env, mixer, team = team, buffer = buffer, episode = episode , N_episodes = N_EPISODES, epsilon_parameters = epsilon_parameters)
        # print(buffer[team].__len__())
       

        # batch = buffer[team].sample(batch_size= BATCH_SIZE)
        while buffer[team].__len__() < BATCH_SIZE:
            generate(env, mixer, team = team, buffer = buffer, episode = episode , N_episodes = N_EPISODES, epsilon_parameters = epsilon_parameters)
            # print(buffer[team].__len__())

        batch  = buffer[team].sample(batch_size = BATCH_SIZE)
        loss = mixer[team].learn(batch)
        print(f'episode {episode}  | loss = {loss} | reward = {cum_reward} | epsilon = {eps}' )










