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


class VDN(nn.Module):
    def __init__(self):
        super(VDN, self).__init__()
        pass


    def eval(self,q_values):
        return torch.sum(q_values, dim = 1)

def criterian(score, target):
    return torch.mean((score - target)**2) 

def eps_greedy(start_eps, NB_EPISODES, episode):
    eps = start_eps
    if eps < 1:
        eps = (NB_EPISODES - episode) / NB_EPISODES 
    else:
        eps = 1

    return eps

NB_EPISODES = 1000
BATCH_SIZE = 40
START_EPS = 0.99 # will decay at each time 
PRINT_EACH  = 100 
MAX_CYCLES = 70
BUFFER_SIZE = 300
GAMMA = 0.1
env = simple_spread_v2.parallel_env(N= 3, max_cycles = MAX_CYCLES)
replay_memory =  Buffer(capacity = BUFFER_SIZE)
env.reset()

# create the different networks
agents_net  = {}
for agent in env.agents:
    agents_net[agent] = DQN(input_nodes = 18, hidden_nodes = 64, output_nodes = 5).to(device)

target_net = copy.deepcopy(agents_net['agent_0']) # will be the same for all the agents 
vdn = VDN().to(device)
NB_EPISODES = 1
for episode in range(NB_EPISODES):
    env.reset()
    state = env.state().ravel()
    eps = START_EPS
    for step in range(MAX_CYCLES-1):
        agents_states = torch.tensor(state.reshape(env.num_agents, -1), device=device)
        q_values = {}
        for idx, agent in enumerate(env.agents):
            q_values[agent] = (agents_net[agent](agents_states[idx]))
            actions = {}
        eps = eps_greedy(eps, NB_EPISODES, episode)
        if random.random() > eps:
            actions  = {agent : int(torch.argmax(q_values[agent]).cpu().detach().numpy()) for agent in env.agents}
            q_chosen = [float(torch.max(q_values[agent])) for agent in env.agents]
        else: 
            actions = {agent: random.choice(range(5)) for agent in env.agents}
            q_chosen = [float(q_values[agent][actions[agent]]) for agent in env.agents]

        observation, reward, is_done, _ = env.step(actions)
        new_state = np.asarray([observation[agent] for agent in env.agents]).reshape(-1)
        action_vect = [actions[agent] for agent in env.agents]
        reward_vect = [reward[agent] for agent in env.agents]
        experience = Experience(state, action_vect, reward_vect, new_state, is_done, q_chosen)
        replay_memory.add(experience)
        state = new_state     
    
    # lets take a sample 
    state_sample, action_sample, reward_sample, new_state_sample , is_done_sample, q_chosen_sample = replay_memory.sample(batch_size = BATCH_SIZE)
    q_chosen_tensor = torch.tensor(q_chosen_sample, device = device)
    scores = vdn.eval(q_chosen_tensor)
    # target = GAMMA * 

        
        
        


"""

print(f"new state {new_state}")
        experience = Experience(state, action_vect, reward_vect, new_state, is_done, q_chosen)
        replay_memory.add(experience)
        state = new_state



for episode in range(NB_EPISODES):
    env.reset()
    cnt =  0
    thrshld = 0

    eps = 0.1
    state = env.state().ravel()
    gamma = 0.1
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

"""