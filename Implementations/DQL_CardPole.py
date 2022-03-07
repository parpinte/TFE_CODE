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
import torchvision.transforms as T
import copy


# define environment 
env = gym.make("CartPole-v0")

# choose device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {device}")

# get the nessacery info for the neural networkx 
input_nodes = len(env.observation_space.low)
print(f"input_nodes = {input_nodes}")
output_nodes = 2
print(f"output_nodes = {output_nodes}")

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




def test_DQN(): 
    agent = DQN().to(device)
    X = 100 * torch.rand(4, device = device)
    print(X)
    print(agent(X))

nbEpisodes = 100
Eps_greedy  = 0.01

policy_net = DQN().to(device)
target_net = copy.deepcopy(policy_net)
# print(target_net)

for episode in range(nbEpisodes):
    state = env.reset()
    q = policy_net(torch.tensor(state, device = device))
    print(q)
    