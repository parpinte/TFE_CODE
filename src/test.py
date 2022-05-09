
import time
import numpy as np
import random

import sys
sys.path.append('..')
import defense_v0 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections

import argparse
import copy
import math
import matplotlib.pyplot as plt


# create graph for epsilon greedy 
import numpy as np
import math
import matplotlib.pyplot as plt

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


class transition():
    def __init__(self,name, net, **kwargs):
        self.name = name
        self.state = kwargs.pop('state', 0)
        self.in_message = kwargs.pop('in_message',0)
        self.action = kwargs.pop('action',0)
        self.reward = kwargs.pop('reward',0)
        self.next_state = kwargs.pop('next_state',0)
        self.next_message = kwargs.pop('next_message', 0)

TERRAIN = 'central_7x7_2v2'
MAX_CYCLES = 200
env = defense_v0.env(terrain= TERRAIN, max_cycles = MAX_CYCLES)
buffer = Buffer(capacity= 1000)

def generate(env, episode, team):
    env.reset()

    for agent in env.agent_iter():
        pass


'''
A = 0.5
B = 0.1
C = 0.7
eps = []
N_episodes = 10000

for episode in range(N_episodes):
    standarized_time = (episode - A * N_episodes)/(B * N_episodes)
    cosh = np.cosh(math.exp(-standarized_time))
    epsilon = 1.1 - (1 /cosh + (episode * C / N_episodes))
    eps.append(epsilon)


episodes = list(range(N_episodes))
plt.plot(episodes, eps, label = f"A = {A}, B = {B}, C = {C}")
# seconde plot 
A = 0.3
B = 0.1
C = 0.7
eps = []
N_episodes = 10000

for episode in range(N_episodes):
    standarized_time = (episode - A * N_episodes)/(B * N_episodes)
    cosh = np.cosh(math.exp(-standarized_time))
    epsilon = 1.1 - (1 /cosh + (episode * C / N_episodes))
    eps.append(epsilon)

plt.plot(episodes, eps, label = f"A = {A}, B = {B}, C = {C}")   

#3rd plot  
A = 0.4
B = 0.2
C = 0.1
eps = []
N_episodes = 10000

for episode in range(N_episodes):
    standarized_time = (episode - A * N_episodes)/(B * N_episodes)
    cosh = np.cosh(math.exp(-standarized_time))
    epsilon = 1.1 - (1 /cosh + (episode * C / N_episodes))
    eps.append(epsilon)

plt.plot(episodes, eps, label = f"A = {A}, B = {B}, C = {C}")   

plt.xlabel("Episodes - axis")
plt.ylabel("epsilon - axis")
plt.title("Epsilon Greedy")
plt.legend(loc="upper right")
plt.show()
'''