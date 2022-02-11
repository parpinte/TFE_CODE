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
# build the network 
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

# gset the different parameters 
input_nodes = len(env.observation_space.low)
print(f"input_nodes = {input_nodes}")
output_nodes = 2
print(f"output_nodes = {output_nodes}")
hidden_nodes = 64
print(f"hidden_nodes = {hidden_nodes}")

eps = 0.9
gamma = 0.01



def main(): 
    agent = DQN().to(device)
    # get the observation 
    state = env.reset()
    stateTensor = torch.tensor(state, device= device)
    is_done = False
    batch = []
    criterian = nn.CrossEntropyLoss()
    optimizer = optim.Adam(agent.parameters(), lr = gamma)
    while is_done == False:
        # get the q values by applying the the agent 
        Qvals = agent(stateTensor)
        # store the different q values in vector 
        Qvals_queue = Qvals.cpu().detach().numpy() # cpu() to copy in memory / detach() leave the grad / numpy() convert to array  
        # evaluate an Eps greedy 
        if random.random() > eps:
            action = np.argmax(Qvals_queue)
            # print(f" no eps = {action}")
        else:
            action = env.action_space.sample()
            # print(f" with eps = {action}")
        
        # take the action 
        New_state, reward, is_done,_ = env.step(action)
        # add the different information to the batch for the training 
        information = (state, action, reward, New_state)
        batch.append(information)
        state = New_state
    print(f"length of the batch = {len(batch)}")

    # Now that the batch is created we need to train our neural network with it 
    # for that different parameters have to be set 
    # 1st isolate the states for the training ( in form of tensor size (batch size , 4 ))
    States_list = [batch[i][0] for i in range(len(batch))]
    # create the tensor for the training
    StatesTensor = torch.tensor(States_list, device=device)
    print(StatesTensor.shape) # tensor.size = (batch_Size, 4) ==> correct format 


    # test if it's fine with the neural network 
    QValues = agent(StatesTensor)
    #  print(QValues)  # it works ==> :) 
    # Store the QValues in a vector (array)
    Q_array = QValues.cpu().detach().numpy()
    # now we need to find which Q was used ( which can be found by looking to the actions taken)
    Actions_taken = [batch[i][1] for i in range(len(batch))]
    print(Actions_taken)
    Qtaken_list = []
    for i, act in enumerate(Actions_taken):
        Qtaken_list.append(Q_array[i,act])
    Q_token = torch.tensor(np.array(Qtaken_list), device=device) 
    Q_token = Q_token[None, :]
    # print(Q_token)
    # create the ground truth formula ( Reward + gamma * max(NN(i)))
    # the rewards are given ( see the batch )
    # NN(S_) is the one we need to find by applying the NN on S_ (in batch )
    # compute max(NN(S_)) 
    States_Tensor = torch.tensor([batch[i][3] for i in range(len(batch))], device = device)
    get_max_QS_ = np.max(agent(States_Tensor).cpu().detach().numpy(),axis = 1)
    Rewards = np.array([batch[i][2] for i in range(len(batch))])
    target = torch.tensor(Rewards  + gamma * get_max_QS_, device=device) # the goal to reach 
    # now we need to compute the lost 
    print(Q_token.shape)
    print(target.shape)
    loss = criterian(Q_token,target)
    






    



if __name__ == '__main__':
    main()