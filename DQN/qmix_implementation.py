# test of q mix 

from pettingzoo.mpe import simple_spread_v2
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
device = 'cuda' # if torch.cuda.is_available()  else "cpu"
print(f"device = {device}")

# buffer configuration 
Experience = collections.namedtuple('Experience',['state','action','reward','new_state','is_done', 'q_chosen'])

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


# generate the buffer 

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
        state, action, reward,  new_state, is_done, q_chosen = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(new_state),
            np.array(is_done),
            np.array(q_chosen)
        )


class QMixer(nn.Module):
    def __init__(self, args):
        super(QMixer, self).__init__()
        # W1 have to be a matrix of nagents * hidden_layer_dim 
        # with pytorch we will create an output of n_agents * hidden_layer_dim and than we will do a reshape 
        self.n_agents = args.n_agents
        self.hidden_layer_dim = args.hidden_layer_dim
        self.state_shape = args.state_shape
        # create the hyper networks 
        self.w1_layer  = nn.Linear(self.state_shape, self.n_agents * self.hidden_layer_dim)
        self.w2_layer  = nn.Linear(self.state_shape, 1 * self.hidden_layer_dim) # size (batch , 1 )
        # consranare 
        self.b1 = nn.Linear(self.state_shape, self.hidden_layer_dim)
        # at b2 we have to get 1 output which is the Qtot 
        self.b2 = nn.Sequential(
                                nn.Linear(self.state_shape, self.hidden_layer_dim),
                                nn.ReLU(),
                                nn.Linear(self.hidden_layer_dim, 1) )   


    # how do we have to do the forward ? 
    def forward(self, states, q_values):
        w1 = torch.abs(self.w1_layer(states))
        
        w2 = torch.abs(self.w2_layer(states)).view(-1, self.hidden_layer_dim, 1 )
        b1 = self.b1(states).view(-1, 1, self.hidden_layer_dim)
        b2 = self.b2(states).view(-1, 1, 1)
        w1 = w1.view(-1, self.n_agents, self.hidden_layer_dim)
        w1b1_out = F.elu(torch.bmm(q_values, w1) + b1)  # F.elu(+ b1) 
        # apply w2
        w2out = torch.bmm(w1b1_out, w2) + b2
        prt = False 
        if prt:
            print(f"w1 output {w1.shape}")
            print(f"w2 output {w2.shape}")
            print(f"b1 reshape {b1.shape}")
            print(f"b2 reshape {b2.shape}")
            print(f"w1b1_out = {w1b1_out.shape}")
            print(f"w2out = {w2out.shape}")
        return w2out




    
max_cycles = 100

env = simple_spread_v2.parallel_env(N= 3, max_cycles=max_cycles)
env.reset()
cnt =  0
thrshld = 0
capacity = 1000
replay_memory =  Buffer(capacity = capacity)
eps = 0.1
state = env.state().ravel()
gamma = 0.01
agents_net  = {}
for agent in env.agents:
    agents_net[agent] = DQN(input_nodes = 18, hidden_nodes = 64, output_nodes = 5).to(device)

def criterian(score, target):
    return torch.mean((score - target)**2) 

NB_EPISODES = 1000
batch_size = 40

def get_target(env, agents_net, qmix, rewards, new_state, gamma, device,batch_size):
    r = torch.tensor(rewards , device = device)
    new_state = torch.tensor(new_state, device = device)
    # need 
    new_agents_states = new_state.reshape(batch_size, env.num_agents, -1)
    # print(f" nAstates = {new_agents_states.shape}")
    q_values = {}
    for id in range(batch_size):
        for idx, agent in enumerate(env.agents):
            q_values[agent] = (agents_net[agent](new_agents_states[id, idx]))

        q_chosen[id] = [float(torch.max(q_values[agent]).cpu().detach().numpy()) for agent in env.agents]  
        q_vals = torch.tensor(q_chosen, device = device, dtype = torch.float32).view(-1, 1,env.num_agents) 
       
        # print(f"q_vals futur {q_vals.shape}")
    qtot_futur = torch.squeeze(qmix(new_state, q_vals))
    target = env.num_agents * r[:,0] + gamma * qtot_futur
    return target 


for episode in range(NB_EPISODES):
    env.reset()
    cnt =  0
    thrshld = 0

    eps = 0.1
    state = env.state().ravel()
    gamma = 0.01
    for step in range(max_cycles -1):
        # get the state of each agent in order to pass them through the network
        agents_states = torch.tensor(state.reshape(env.num_agents, -1), device=device)
        # print(f"agent state {agents_states.shape}")
        # need the q values using DQN 
        q_values = {}
        for idx, agent in enumerate(env.agents):
            q_values[agent] = (agents_net[agent](agents_states[idx]))
        # from thi s we only need 1 value for earch action 
        actions = {}
        if random.random() > eps:
            actions  = {agent : int(torch.argmax(q_values[agent]).cpu().detach().numpy()) for agent in env.agents}
            q_chosen = [float(torch.max(q_values[agent]).cpu().detach().numpy()) for agent in env.agents]
        else: 
            actions = {agent: random.choice(range(5)) for agent in env.agents}
            q_chosen = [float(q_values[agent][actions[agent]]) for agent in env.agents]

        observation, reward, is_done, _ = env.step(actions)
        action_vect = [actions[agent] for agent in env.agents]
        reward_vect = [reward[agent] for agent in env.agents]
        new_state = np.concatenate([observation[agent] for agent in env.agents]).flatten()
        experience = Experience(state, action_vect, reward_vect, new_state, is_done, q_chosen)
        replay_memory.add(experience)
        state = new_state

        # get Qtot value 
        # 1) sample in the buffer
    # Q tot 
    # set the parameters

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_agents", default = 3)
    parser.add_argument("--hidden_layer_dim",default  = 130)
    parser.add_argument("--state_shape", default = 54)
    args = parser.parse_args()
    batch_size = 40
    # optimizer = optim.Adam(agent.parameters(), lr = gamma)
    qmix= QMixer(args).to(device)
    state, action, reward, new_state , is_done, q_chosen= replay_memory.sample(batch_size = batch_size)

    # print(q_chosen.shape)

    state = torch.tensor(state, device = device)
    # print(state.shape)
    q_vals = torch.tensor(q_chosen, device = device, dtype = torch.float32).view(-1, 1,env.num_agents)


    # print(f"state in = {state.shape}")
    # print(f"q_vals = {q_vals.shape}")
    scores = torch.squeeze(qmix(state, q_vals)) # it works (y) 
    # target = reward[:,0] + torch.tensor( , device = device)


    # need to compute the loss 
    # have to remember that the reward given is 
 




    # print(get_target(env, agents_net, qmix, reward, new_state, gamma, device, batch_size))


    target = get_target(env, agents_net, qmix, reward, new_state, gamma, device, batch_size)
    loss = criterian(scores, target)
    # print(loss)
    parameters = list(agents_net['agent_0'].parameters())
    parameters += qmix.parameters()
    optimizer = optim.Adam(parameters, lr = gamma)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for agent in env.agents:
        agents_net[agent] = copy.deepcopy(agents_net['agent_0'])






def demo():
    env.reset()
    state = env.state().ravel()
    eps = 0
    for step in range(max_cycles -1):
        
        
        # get the state of each agent in order to pass them through the network
        agents_states = torch.tensor(state.reshape(env.num_agents, -1), device=device)
        # print(f"agent state {agents_states.shape}")
        # need the q values using DQN 
        q_values = {}
        for idx, agent in enumerate(env.agents):
            q_values[agent] = (agents_net[agent](agents_states[idx]))
        # from thi s we only need 1 value for earch action 
        actions = {}
        if random.random() > eps:
            actions  = {agent : int(torch.argmax(q_values[agent]).cpu().detach().numpy()) for agent in env.agents}
            q_chosen = [float(torch.max(q_values[agent]).cpu().detach().numpy()) for agent in env.agents]
        else: 
            actions = {agent: random.choice(range(5)) for agent in env.agents}
            q_chosen = [float(q_values[agent][actions[agent]]) for agent in env.agents]

        
        observation, reward, is_done, _ = env.step(actions)
        env.render()
        # print(observation)
        new_state = np.concatenate([observation[agent] for agent in env.agents]).flatten()
        state = new_state
        time.sleep(0.01)


demo()