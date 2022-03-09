#### The aim of this part is to add a replay meemory ( buffer) 

"""
1) initialize replay memory capacity 
2) initialize the network with random weights 
3) for each episode : 
    1 - initialize the starting state
    2 - For each time step : 
        1) select ann action (exploration / exploitation )
        2) Execute the chosen action 
        3) observe the reward and next state
        4) store the experience in relay memory 
            => Experience is : e(t) = (s,a,r_,s_)
            (state, action, reward, new_state)
"""


from re import M
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torchvision.transforms as T
import collections

# define environment 
env = gym.make("CartPole-v0")

# choose device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")

# build the network 
class DQN(nn.Module):
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(DQN, self).__init__()
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.linear = nn.Sequential(
            nn.Linear(self.input_nodes, self.hidden_nodes),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes, self.output_nodes),
        )

    def forward(self, x):
        output = self.linear(x)
        return output


# Define a class buffer 
# lets determine a named tuple wich contain all the different information needed for# as esplained in the intrduction 
# the named tuple will be the experience 
Experience = collections.namedtuple('Experience',['state','action','reward','new_state'])

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
        state, action, reward, new_state = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(new_state)
        )
        
# loss function 
def criterian(score, target):
    return torch.mean((score - target)**2) 
# optimize model 
def get_experience(env, agent, replay_memory, is_done):
    
    state = env.reset()
    while is_done == False:
        state_tensor = torch.tensor(state, device= device)
        q_values  = agent(state_tensor)
        q_values_array  = q_values.cpu().detach().numpy()
        # choose the action 
        if random.random() > eps:
            action = np.argmax(q_values_array)
        else: 
            action = env.action_space.sample()
            
        # execute the action 
        new_state, reward, is_done, _ = env.step(action)
        # need to create an experience 
        """
        ['state','action','reward','new_state']
        """
        experience = Experience(state, action, reward, new_state)
        # add the experience to the buffer 
        replay_memory.add(experience)
        state = new_state




# gset the different parameters 
input_nodes = len(env.observation_space.low)
print(f"input_nodes = {input_nodes}")
output_nodes = 2
print(f"output_nodes = {output_nodes}")
hidden_nodes = 64
print(f"hidden_nodes = {hidden_nodes}")

eps = 0.9
gamma = 0.01

nb_episodes = 4000
BUFFER_SIZE = 10000
BATCH_SIZE = 64

replay_memory = Buffer(BUFFER_SIZE)
def main():
    agent = DQN(input_nodes, hidden_nodes, output_nodes).to(device)
    for _ in range(nb_episodes):

        is_done = False
        optimizer = optim.Adam(agent.parameters(), lr = gamma)

        get_experience(env, agent, replay_memory, is_done)
        
        while BATCH_SIZE > replay_memory.__len__(): 
            get_experience(env, agent, replay_memory, is_done)
        # train the neural network 
        # get a batch for the training 
        states, rewards, actions, new_states = replay_memory.sample(BATCH_SIZE)
        # print(states)
        # print(rewards)
        # print(actions)
        # print(new_states)
        states = torch.tensor(states, device = device)
        q_values = agent(states)
        q_token = q_values[range(len(actions)), actions]
        new_states = torch.tensor(new_states, device = device)
        get_max_QS_ = torch.max(agent(new_states), axis = 1)[0]
        reward = get_max_QS_ + gamma * torch.tensor(rewards, device = device)
        loss = criterian(q_token, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

 



if __name__ == '__main__':
    main()
    
