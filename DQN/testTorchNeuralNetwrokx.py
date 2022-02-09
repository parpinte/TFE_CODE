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
    def __init__(self, *kargs):
        super(DQN, self).__init__()
        self.input_nodes = 4
        self.hidden_nodes_1 = 64
        self.hidden_nodes_2 = 32
        self.output_nodes = 2
        self.linear = nn.Sequential(
            nn.Linear(self.input_nodes, self.hidden_nodes_1),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes_1, self.hidden_nodes_2),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes_2, self.output_nodes),
        )

    def forward(self, x):
        output = self.linear(x)
        return output




def test_DQN(): 
    agent = DQN().to(device)
    x = torch.tensor(env.reset(), device= device)
    print(x)
    print(agent(x))
    print(torch.argmax(agent(x)))
test_DQN()