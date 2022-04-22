################################### Qmix at defense v0 ###############################""""
from re import S
from ray import get

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

################################# Qmixer class  #######################""

class QMixer(nn.Module):
    def __init__(self, n_agents, hidden_layer_dim, state_shape, **kwargs):
        super(QMixer, self).__init__()
        # W1 have to be a matrix of nagents * hidden_layer_dim 
        # with pytorch we will create an output of n_agents * hidden_layer_dim and than we will do a reshape 
        self.n_agents = n_agents
        self.hidden_layer_dim = hidden_layer_dim
        self.state_shape = state_shape
        self.ELU = nn.ELU()
        # create the hyper networks 
        self.w1_layer  = nn.Linear(self.state_shape, self.n_agents * self.hidden_layer_dim)
        self.w2_layer  = nn.Linear(self.state_shape, 1 * self.hidden_layer_dim) # size (batch , 1 )
        self.p = kwargs.get('dropout', 0.25)
        self.dropout = nn.Dropout(p = self.p)
        # consranare 
        self.b1 = nn.Linear(self.state_shape, self.hidden_layer_dim)
        # at b2 we have to get 1 output which is the Qtot 
        self.b2 = nn.Sequential(
                                nn.Dropout(p = self.p),
                                nn.Linear(self.state_shape, self.state_shape),   # , self.hidden_layer_dim
                                nn.ReLU(),
                                nn.Dropout(p = self.p),
                                nn.Linear(self.state_shape, 1)  # self.hidden_layer_dim, 1
                                )   


    # how do we have to do the forward ? 
    def forward(self, states, q_values):
        states = self.dropout(states)
        w1 = torch.abs(self.w1_layer(states))
        w1 = w1.reshape(-1, self.n_agents, self.hidden_layer_dim)

        w2 = torch.abs(self.w2_layer(states))
        w2 = w2.reshape(-1,self.hidden_layer_dim, 1)

        b1 = self.b1(states).reshape((-1, 1, self.hidden_layer_dim))

        b2 = self.b2(states).reshape((-1, 1, 1))

        out1 = self.ELU(torch.add(torch.bmm(q_values,w1), b1))
        Qtot = torch.add(torch.bmm(out1, w2), b2)
        Qtot = Qtot.reshape(-1)

        return Qtot


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
        state, action, reward,  new_state, is_done, action_mask = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(new_state),
            np.array(is_done),
            np.array(action_mask),
        )
    def clear(self):
        self.buffer.clear()

######################## agent class ( DQN but it can be a GRU ) #############################

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
            nn.Linear(self.hidden_nodes, self.output_nodes),
        )

    def forward(self, x):
        output = self.linear(x)
        return output


"""

une fois on a enregistré la derniere episode 

"""
################################### VDN ############################
class VDN():
    def __init__(self):
        pass
    def forward(self):
        pass



######################################## Agent ######################################

class Agent():
    def __init__(self, name, net, **kwargs):
        self.name = name
        self.net = net
        self.target_net = copy.deepcopy(self.net)
        self.target_net.eval()
        self.device = kwargs.get('device','cuda')
        self.epsilon = 1.0
        

    def set_epsilon(self, N_episodes, episode, parameters):
        A = parameters.A
        B = parameters.B
        C = parameters.C
        standarized_time = (episode - A * N_episodes)/(B * N_episodes)
        cosh = np.cosh(math.exp(-standarized_time))
        epsilon = 1.1 - (1 /cosh + (episode * C / N_episodes))

        self.epsilon = epsilon

    def get_action(self, state, done):
        s = state['obs']
        action_mask = state['action_mask']
        
        if random.random() < self.epsilon:
            p = action_mask / np.sum(action_mask)
            action = int(np.random.choice(range(len(p)), p = p)) if not done else None
        else: 
            with torch.no_grad():
                state_tensor = torch.tensor(s, device = self.device, dtype= torch.float32)
                qvals = self.net(state_tensor)
                mask_tensor = torch.tensor(action_mask, dtype = torch.bool)
                qvals[~ mask_tensor] = -100.0
                # print('q vals is equal to ########################################### \n')
                # print(qvals)
                # print('action mask is ###########################################')
                # print(~ mask_tensor)        
                action = qvals.cpu().squeeze().argmax().item() if not done else None


        return action
    def target_action(self, state, done):
        s = state['obs']
        action_mask = state['action_mask']
        
        with torch.no_grad():
            state_tensor = torch.tensor(s, device = self.device, dtype= torch.float32)
            qvals = self.net(state_tensor)
            mask_tensor = torch.tensor(action_mask, dtype = torch.bool)
            qvals[~ mask_tensor] = -100.0
            # print('q vals is equal to ########################################### \n')
            # print(qvals)
            # print('action mask is ###########################################')
            # print(~ mask_tensor)        
            action = qvals.cpu().squeeze().argmax().item() if not done else None


        return action


    def sync(self):
        self.target_net.load_state_dict(self.net.state_dict())

    



#  The Mixer 
class Mixer:
    def __init__(self, env, team, device= 'cpu', mixer = 'Qmix', **kwargs):
        # need (agents , observation space , action space and state space in order to define all the nn ( agents + mixer )
        self.agent_names = copy.copy(team)
        self.n_agents = len(self.agent_names)
        self.state_shape = env.observation_space(self.agent_names[0])['obs'].shape[0]
        self.action_space = env.action_space(self.agent_names[0]).n
        self.use_mixer = mixer
        self.epsilon = kwargs.get('epsilon',0.99)
        self.device = device
        self.dropout = kwargs.get('dropout', 0.25)
        # determine the different agents networkx
        self.agents = {}
        self.agent_net = DQN(input_nodes = self.state_shape, hidden_nodes = 64, output_nodes = self.action_space, dropout = self.dropout).to(self.device)
        # define the different agents 
        for agent in self.agent_names:
            self.agents[agent] = Agent(agent, self.agent_net, epsilon = self.epsilon, device = self.device, dropout = self.dropout)
        
        # ask which mixer we need to use 
        
        if self.use_mixer == 'Qmix':
            self.net = QMixer(n_agents = self.n_agents, hidden_layer_dim = 64, state_shape = self.n_agents * self.state_shape, dropout = self.dropout).to(self.device)
        elif self.use_mixer == 'VDN':
            self.net = VDN()

        # target net 
        self.target_net = copy.deepcopy(self.net)
        self.target_net.eval()

        # set the different parameters for the optimization 
        parameters = list(self.net.parameters())
        # add the parameters of the agent network 
        parameters += self.agent_net.parameters()

        # define the optimizer which will be used for the learning 
        self.lr = kwargs.get('lr', 0.0001)
        self.gamma = kwargs.get('gamma',.9)

        self.optimizer = torch.optim.RMSprop(parameters, self.lr)

        # loss function 
        self.MSE = nn.MSELoss()

    def sync(self):
        self.target_net.load_state_dict(self.net.state_dict()) 
        # give order to sync for the different agents 
        for agent in self.agents.values():
            agent.sync()

    def epsilon_order(self,N_episodes, episode, parameters_epsilon):
        for agent in self.agents.values():
            agent.set_epsilon(N_episodes, episode, parameters_epsilon)
        return self.agents[self.agent_names[0]].epsilon


    def get_q_values(self, batch):
        # batch form = 'state','action','reward','new_state','is_done'
        # batch shape (BATCH_SIZE, dictionnary) 
        # we need just Qagents (BATCH_SIZE, n_agents)
        # we will first get only the observations which is the first dimension 
        state = batch[0] #  (batch size, dictionnary of the state of each agent )
        len_s = state.size
        #
        Q = []
        for agent in self.agent_names:
            s = [torch.tensor(state[idx][agent], device = self.device, dtype = torch.float32).unsqueeze(0) for idx in range(len_s)]  # create the batch that will pass through the network  
            s = torch.cat(s)
            Q.append(self.agents[agent].net(s).unsqueeze(1))

        q_vals = torch.cat(Q, dim = 1)
        return q_vals

    def get_target_qvalues(self, batch):
        # the target q values will be the one to compute the target in order to compute the loss 
        state_ = batch[3]    # (batch size, dictionnary ) 
        len_s_ = state_.size

        Q_target = []
        for agent in self.agent_names:
            s_ = [torch.tensor(state_[idx][agent], device = self.device, dtype = torch.float32).unsqueeze(0) for idx in range(len_s_)]  # create the batch that will pass through the network  
            s_ = torch.cat(s_) 
            Q_target.append( self.agents[agent].target_net(s_).unsqueeze(1))

        q_target_vals = torch.cat(Q_target, dim = 1)
        return q_target_vals  

    def update_agent(self):
        for agent in self.agent_names:
            self.agents[agent].net.load_state_dict(self.agent_net.state_dict()) 


    def concat(self, dict):
        keys = dict.keys()
        elements = [dict[key] for key in keys]
        elements = np.concatenate(elements)
        return elements  


    def learn(self, batch):
        # what we need for the learning function batch
        # curret state that need to be concatenated for the qmix_net 
        s = batch[0] 
        size_s = s.size
        s = [torch.tensor(self.concat(s[idx]), device = self.device, dtype = torch.float32).unsqueeze(0) for idx in range(size_s)]
        s = torch.cat(s) # ==> give tensor (batchSize, 54 ( 3 agents ))

        # the state 
        s_ = batch[3]
        size_s_ = s_.size
        s_ = [torch.tensor(self.concat(s_[idx]), device = self.device, dtype = torch.float32).unsqueeze(0) for idx in range(size_s_)]
        # print('\n \n')
        # print(s_)
        s_ = torch.cat(s_) # ==> give tensor (batchSize, 54 ( 3 agents ))


        # get the qvalues 
        Qvalues = self.get_q_values(batch)
        
        
        actions = batch[1]
        actions = [list(actions[idx].values()) for idx in range(size_s)]
        q = np.zeros((size_s, self.n_agents))
        with torch.no_grad():
            for idx in range(len(actions)):
                for index in range(len(actions[idx])):
                    if (actions[idx][index] is None):
                        q[idx, index] = 0.0
                    else:
                        q[idx, index] = Qvalues[idx, index, actions[idx][index]].item()
                        # print(q[idx, index])
        
        q_values = torch.tensor(q, device = self.device, dtype = torch.float32).unsqueeze(1)
        
        #actions = torch.cat([torch.tensor(np.array([actions[idx][agent] for agent in self.agent_names]), device = self.device).unsqueeze(0) for idx in range(actions.size)])
        #actions  = actions.unsqueeze(2)
        #q_values = Qvalues.gather(2,actions).squeeze(2).unsqueeze(1) # .squeeze(2) # need  to be verified 
        # get the q tot 
        Qtot = self.net(s, q_values).squeeze()

        # get the reward 

        reward = batch[2]
        reward = torch.tensor(np.array([reward[idx][self.agent_names[0]] for idx in range(reward.size)]), dtype= torch.float32, device = self.device)

        # is done needed 
        isdone = batch[4]
        isdone =  [isdone[idx][self.agent_names[0]] for idx in range(isdone.size)]
        isdone = torch.tensor(isdone, device = self.device)

        # update the Qtot 
        Qtot[isdone] = 0
        # no grad needed here so 
        with torch.no_grad():
            # get the q target 
            q_target_agents = torch.max(self.get_target_qvalues(batch), dim = 2)[0].unsqueeze(1)
            qtot_target  = self.target_net(s_, q_target_agents).squeeze()
            target  = reward + self.gamma * qtot_target.detach()
            # print(qtot_target)
        
        # compute the loss 
        loss = self.MSE(target, Qtot)
        # print(target)
        # loss = torch.tensor(loss.item())
        # loss.requires_grad = True
        # back prop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
################ EPSILON DECAY #########################
class epsilon_params():
    def __init__(self, A = 0.3, B = 0.1, C = 0.1):
        self.A = A
        self.B = B
        self.C = C
####################################### Tracker #####################################
class Tracker():
    def __init__(self, agents):
        self.agents = agents
    
    def next_agent(self, agent):
        pass
####################################### Runner #####################

class Runner():
    def __init__(self, configuration_file):
        with open(configuration_file) as file:
            self.params = yaml.load(file, Loader = yaml.FullLoader)
        self.TERRAIN = self.params['TERRAIN']
        self.MAX_CYCLES = self.params['MAX_CYCLES']

        # simulation 
        self.N_EPISODES = self.params['N_EPISODES']
        self.EPSILON_START = self.params['EPSILON_START']
        self.UPDATE_TIME = self.params['UPDATE_TIME']
        self.MIXER_TYPE = self.params['MIXER_TYPE']
        self.DEVICE  = self.params['DEVICE']

        # EPSILON
        self.EPS = epsilon_params(self.params['A'],self.params['B'],self.params['C'])



        # BUFFER PARAMETRS
        self.CAPACITY = self.params['CAPACITY']
        self.BATCH_SIZE = self.params['BATCH_SIZE']


        # NETWORKX
        self.LEARNING_RATE = self.params['LEARNING_RATE']
        self.GAMMA = self.params['GAMMA']
        self.DROPOUT = self.params['DROPOUT']

        self.env = defense_v0.env(terrain= self.TERRAIN, max_cycles = self.MAX_CYCLES)
        self.env.reset()
        self.blue_team, self.red_team = self.teams_creation()
        self.Experience = collections.namedtuple('Experience',['state','action','reward','new_state','is_done','action_mask'])
        self.set_mixer() 
        self.buffer = Buffer(capacity= self.CAPACITY)
        self.Trainning_team = 'blue'

    def teams_creation(self):
        blue_team = []
        red_team = []
        for agent in self.env.agents:
            if 'blue' in agent:
                blue_team.append(agent)
            else: 
                red_team.append(agent)
        return blue_team, red_team

    def set_mixer(self):
        self.blue_mixer = Mixer(env = self.env, team = self.blue_team, device= self.DEVICE,
                     mixer = self.MIXER_TYPE, dropout = self.DROPOUT, lr = self.LEARNING_RATE, gamma = self.GAMMA)
        self.red_mixer = Mixer(env = self.env, team = self.red_team, device= self.DEVICE,
                     mixer = self.MIXER_TYPE, dropout = self.DROPOUT, lr = self.LEARNING_RATE, gamma = self.GAMMA)
    
    def generate_episode(self, training_team, episode):
            
        self.env.reset() 
        n = int(len(self.env.agents) / 2)
        # which information we want to store ? 
        
        s = []
        a = []
        re = []
        d = []
        am = []
        
        # 
        cum_reward = 0.0
        state = {}
        actions = {}
        reward = {}
        is_done = {}
        action_mask = {}

        for agent in self.env.agent_iter():
            last_agent = self.env.agents[-1]
            
            
            obs, r, done, _ = self.env.last()
            if training_team in agent:
                state[agent] = obs['obs']
                action_mask[agent] = obs['action_mask']
                reward[agent] = r
                is_done[agent] = done
                if training_team == 'blue':
                    self.blue_mixer.epsilon_order(self.N_EPISODES, episode, self.EPS)
                    if self.blue_mixer.agents[agent].epsilon < 0.01:
                        self.blue_mixer.agents[agent].epsilon = 0.01
                    action = self.blue_mixer.agents[agent].get_action(obs, done)
                    actions[agent] = action
                    
                else: 
                    self.red_mixer.epsilon_order(self.N_EPISODES, episode, self.EPS)
                    if self.red_mixer.agents[agent].epsilon < 0.01:
                        self.red_mixer.agents[agent].epsilon = 0.01
                    action = self.red_mixer.agents[agent].get_action(obs, done)
                    actions[agent] = action


            else:
                if 'blue'in agent:
                    self.blue_mixer.agents[agent].epsilon = 1
                    action = self.blue_mixer.agents[agent].get_action(obs, done)
                    
                else: 
                    self.red_mixer.agents[agent].epsilon = 1
                    action = self.red_mixer.agents[agent].get_action(obs, done)
            # self.env.render()
            # time.sleep(1)
            self.env.step(action)
                # experience = self.Experience(copy.deepcopy(state), copy.deepcopy(actions), copy.deepcopy(reward), 
                                    #copy.deepcopy(new_state), copy.deepcopy(is_done), copy.deepcopy(action_mask))
            
            
            #
            if agent == last_agent:
                s.append(copy.copy(state))
                a.append(actions)
                re.append(reward)
                d.append(is_done)
                am.append(action_mask)
                state = {}
                actions = {}
                reward = {}
                is_done = {}
                action_mask = {}
            
            if (training_team in agent) & (agent == 'blue_0'):
                cum_reward += r
                epsilon = self.blue_mixer.agents['blue_0'].epsilon
            else:
                if (training_team in agent) & (agent == 'red_0'):
                    cum_reward += r
                    epsilon = self.red_mixer.agents['red_0'].epsilon
            
        self.tracker(Training_team = 'blue', l = s, replace = True)
        self.tracker(Training_team = 'blue', l = a, replace = True)
        self.tracker(Training_team = 'blue', l = re, replace = True)
            
        self.tracker(Training_team = 'blue', l = d, replace = True)
        self.tracker(Training_team = 'blue', l = am, replace = True)
        
        for index in range(len(s) - n):
            experience = self.Experience(copy.deepcopy(s[index]), copy.deepcopy(a[index]), copy.deepcopy(re[index + n]), 
                                    copy.deepcopy(s[index + n]), copy.deepcopy(d[index]), copy.deepcopy(am[index]))
            self.buffer.add(experience)
        
        return epsilon, cum_reward
    
    def tracker(self, Training_team, l, replace):
        if Training_team == 'blue':
            team = self.blue_team
        else: 
            team = self.red_team
        for index, transition in enumerate(l):
            for agent in team:
                if not (agent in transition):
                    # print(f'{agent} is not in the list at index {index}')
                    if replace: 
                        transition[agent] = l[index - 1][agent]
                    else: 
                        transition[agent] = l[index - 1][agent]
                        
        


    def train(self, training_team):
        # restart buffer
        self.buffer.clear()
        writer = SummaryWriter('src/runs')
        for episode in range(self.N_EPISODES):
            eps, cum_reward = self.generate_episode(training_team = training_team, episode = episode)
            while self.buffer.__len__() < self.BATCH_SIZE: 
                eps, cum_reward = self.generate_episode(training_team = training_team, episode = episode)

            # create the writer 
            batch = self.buffer.sample(self.BATCH_SIZE)

            if training_team == 'blue':
                loss = self.blue_mixer.learn(batch)
            else:
                loss = self.red_mixer.learn(batch)

            writer.add_scalar('reward', cum_reward, episode)
            writer.add_scalar('loss', loss, episode)
            writer.add_scalar('epsilon', eps, episode)
            if episode % self.UPDATE_TIME == 0: 
                self.blue_mixer.sync()
                self.red_mixer.sync()
                print(f'episode {episode} | loss = {loss} | cum_reward = {cum_reward} | eps = {eps} ')

            

    def demo(self):
        self.env.reset() 
        last_agent = self.env.agents[-1]
        cum_reward = 0.0
        for agent in self.env.agent_iter():
            obs, r, done, _ = self.env.last()
            if 'blue' in agent:
                self.blue_mixer.agents[agent].epsilon = 0
                action = self.blue_mixer.agents[agent].target_action(obs, done)
            else: 
                self.red_mixer.agents[agent].epsilon = 0
                action = self.red_mixer.agents[agent].target_action(obs, done)
            self.env.step(action)
            if agent == last_agent:
                cum_reward += r
            self.env.render()

    def _str_(self):
        txt = "**** defense v0 simulation **** \n"
        txt += f"Used Terrain = {self.TERRAIN} \n"
        txt += f"Used device = {self.DEVICE} \n"
        txt += f"Maximum cycles = {self.MAX_CYCLES}  \n"
        txt += f"Team blue => {self.blue_team} \n"
        txt += f"Team blue => {self.red_team} \n"
        txt += f"Number of episodes = {self.N_EPISODES} \n"
        txt += f"UPDATE each {self.UPDATE_TIME} episodes \n"
        txt += f"Mixer type = {self.MIXER_TYPE} \n"
        txt += f"epsilon greedy an exponential \n"
        txt += f"Buffer size = {self.CAPACITY} \n"
        txt += f"Batch size = {self.BATCH_SIZE} \n"
        txt += f"Learning rate = {self.LEARNING_RATE} \n"
        txt += f"Gamma = {self.GAMMA} \n"
        txt += "******************************************"
        txt += f"\n \n \n Start of Simulation ... \n"
        print(txt)
                




if __name__ == '__main__':
    configuration_file = r'configuration.yaml'
    demo = 'y'
    runner = Runner(configuration_file)
    runner._str_()
    epsilon, cum_reward = runner.generate_episode('blue', 1)
    batch = runner.buffer.sample(runner.buffer.__len__())
    print(batch[2])
    a = 1

    """
    runner.train('blue')
    demo = input("Do you want to run a demo ? ")
    while demo == 'y':

        runner.demo()
        demo = input("Do you want to run a demo ? ")
"""