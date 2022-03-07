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

import matplotlib.pyplot as plt


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
            nn.Linear(self.hidden_nodes, self.hidden_nodes),
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
    def end_element(self):
        return self.buffer[-1]


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

def get_target(env, agents_net, reward_sample, new_state_sample, GAMMA, device, BATCH_SIZE):
    r = torch.sum(torch.tensor(reward_sample , device = device), dim =1)
    # r = torch.tensor(reward_sample , device = device)
    new_state_sample = torch.tensor(new_state_sample, device = device)
    # need 
    # print(new_state_sample.shape)
    new_agents_states = new_state_sample.reshape(BATCH_SIZE, env.num_agents, 18)
    
    # print(new_agents_states.shape)
    q_vals = []
    q_values = {}
    for id in range(BATCH_SIZE):
        for idx, agent in enumerate(env.agents):
            q_values[agent] = (agents_net(new_agents_states[id, idx]))
        qs = np.array(([q_values[agent].max().cpu().detach().numpy() for agent in env.agents]))
        
        q_vals.append(qs)
    qvals = np.array(q_vals)
    q_max_futur = torch.max(torch.tensor(q_vals, device=device, dtype= torch.float32), dim =1)[0]
    target =  r / env.num_agents + GAMMA * q_max_futur
    return target

def demo(L = 20):
    for _ in range(L):
        env.reset()
        state = env.state().ravel()
        for step in range(MAX_CYCLES-1):
            agents_states = torch.tensor(state.reshape(env.num_agents, -1), device=device)
            q_values = {}
            for idx, agent in enumerate(env.agents):
                q_values[agent] = (target_net(agents_states[idx]))
                actions = {}
            
            actions  = {agent : int(torch.argmax(q_values[agent])) for agent in env.agents}
            q_chosen = [float(torch.max(q_values[agent])) for agent in env.agents]


            observation, reward, is_done, _ = env.step(actions)
            new_state = np.asarray([observation[agent] for agent in env.agents]).reshape(-1)
            env.render()
            time.sleep(0.01)
            state = new_state     




if __name__ == '__main__':
    NB_EPISODES = 2000
    BATCH_SIZE = 10
    START_EPS = 0.99 # will decay at each time 
    PRINT_EACH  = 100 
    UPDATE_EACH = 50
    MAX_CYCLES = 101
    BUFFER_SIZE = 300
    GAMMA = 0.01
    LEARNING_RATE = 0.9
    env = simple_spread_v2.parallel_env(N= 3, max_cycles = MAX_CYCLES)
    replay_memory =  Buffer(capacity = BUFFER_SIZE)
    env.reset()

    # create the different networks
    agents_net = DQN(input_nodes = 18, hidden_nodes = 64, output_nodes = 5).to(device)

    target_net = copy.deepcopy(agents_net) # will be the same for all the agents
     
    vdn = VDN().to(device)

    parameters = agents_net.parameters()
    optimizer = optim.Adam(parameters, lr = LEARNING_RATE)


    Rtot = np.zeros(NB_EPISODES)
    for episode in range(NB_EPISODES):
        env.reset()
        state = env.state().ravel()
        eps = START_EPS
        cumul_reward = 0
        for step in range(MAX_CYCLES-1):
            agents_states = torch.tensor(state.reshape(env.num_agents, -1), device=device)
            q_values = {}
            for idx, agent in enumerate(env.agents):
                q_values[agent] = (agents_net(agents_states[idx]))
                actions = {}
            eps = eps_greedy(eps, NB_EPISODES, episode)
            if random.random() > eps:
                actions  = {agent : int(torch.argmax(q_values[agent])) for agent in env.agents}
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
            # print(f"Stat")
            cumul_reward +=  env.num_agents * reward['agent_0']    

        # lets take a sample 
        state_sample, action_sample, reward_sample, new_state_sample , is_done_sample, q_chosen_sample = replay_memory.sample(batch_size = BATCH_SIZE)
        q_chosen_tensor = torch.tensor(q_chosen_sample, device = device)
        scores = vdn.eval(q_chosen_tensor)
        
        # get target 

        target = get_target(env, agents_net, reward_sample, new_state_sample, GAMMA, device, BATCH_SIZE)
        

        loss = criterian(scores, target)
        loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # get the last reward 
        Rtot[episode] = cumul_reward
        if episode % PRINT_EACH == 0:
            print(f"episode = {episode}, reward = {Rtot[episode]}, EPS = {eps}, loss = {loss}, buffer size = {replay_memory.__len__()}")
        
        if episode % UPDATE_EACH == 0:
            target_net.load_state_dict(agents_net.state_dict)
            # target_net = copy.deepcopy(agents_net)

    ep = range(NB_EPISODES)
    plt.plot(ep, Rtot)
    plt.savefig('reward.png')
    demo()



  


