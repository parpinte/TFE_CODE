""" Independent Q-learning for DEFENSE (with action mask) """

import random
import copy
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from utilities import Logger, build_network

# hack to allow import of env
import sys; sys.path.insert(0, '.')
from env import defense_v0

# %%
LOG_INTERVAL = 1
SYNC_RATE = 10
optimizer = optim.Adam 

DEFAULT_OBS = np.zeros(8)

# %%
class EpisodeStep:
    def __init__(self, observation, mask, action, reward, done, next_obs, next_mask):
        self.observation = observation
        self.mask = mask
        self.action = action
        self.reward = reward
        self.done = done
        self.next_obs = next_obs
        self.next_mask = next_mask
        self.counter = 0
    
    def __iter__(self):
        all = [self.observation, self.mask, self.action, self.reward, self.done, self.next_obs, self.next_mask]
        return iter(all)

# %%
class Runner:
    """
    Implementation of a simple DQN algorithm.
    Algorithm:
        1. Collect some examples by acting in the environment and store them in a replay memory
        |   2. Every K steps sample N examples from replay memory
        |   |   3. For each example calculate the target (bootstrapped estimate of the discounted value of the state and action taken), y, using a neural network to approximate the Q function. s' is the next state following the action actually taken.
        |   |           y_t = r_t + gamma * argmax_a Q(s_t', a)
        |   |   4. For each example calculate the current estimate of the discounted value of the state and action taken
        |   |           x_t = Q(s_t, a_t)
        |   |   5. Calculate L(x, y) where L is a regression loss (eg. mse)
        |   |   6. Calculate the gradient of L with respect to all the parameters in the network and update the network parameters using the gradient
        |   7. Repeat steps 3 - 6 M times
        8. Repeat steps 2 - 7 Z times
        9. Repeat steps 1 - 8

    For more information on Q-Learning see Sergey Levine's lectures 6 and 7 from CS294-112 Fall 2017
    https://www.youtube.com/playlist?list=PLkFD6_40KJIznC9CDbVTjAF2oyt8_VAe3
    """
    def __init__(self, **kwargs):
        self.episode_length = kwargs.get('episode_length', 200)
        self.max_distance = kwargs.get('range', 5)
        self.env = defense_v0.env(terrain='flat_5x5', max_cycles=self.episode_length, max_distance=self.max_distance )
        self.env.reset()

        self.agents, self.buffers = {}, {}
        # select which agents learn and wich don't (= all others)
        if kwargs['learners'] == 'all':
            self.learners = self.env.agents[:] # ['blue_0'] #   ['blue_0'] #  ['adversary_0'] #
        elif  kwargs['learners'] == 'blue':
            self.learners = ['blue_0']
        elif kwargs['learners'] == 'red':
            self.learners = ['red_0']
        self.others   = [agent for agent in self.env.agents if agent not in self.learners]
        for agent in self.env.agents:
            if agent in self.learners:
                dims = self.env.observation_spaces[agent]['obs'].shape[0]
                self.agents[agent] = Agent(name=agent,
                                        #dims=self.env.observation_spaces[agent].shape[0],
                                        dims=dims,
                                        actions=self.env.action_spaces[agent].n,
                                        gamma=kwargs.get('gamma', 0.99),
                                        lr=kwargs.get('lr', 0.001),
                                        layers=kwargs.get('layers', [128, 128]))
                self.buffers[agent] = deque(maxlen=kwargs.get('buffer_size', 1024))
            elif agent in self.others:
                #self.agents[agent] = StaticAgent(name=agent)
                n_actions = self.env.action_spaces[agent].n
                self.agents[agent] = RandomAgent(name=agent, n_actions=n_actions)

        self.epsilon = 1.0
        self.eps_min = kwargs.get('eps_min', 0.1)
        self.eps_decay = (self.epsilon - self.eps_min)/kwargs.get('eps_steps', 20000)
        self.n_batches = kwargs.get('n_batches', 32)     # K steps
        self.sample_size = kwargs.get('sample_size', 64) # N examples from replay memory
        self.sync_rate = kwargs.get('sync_rate', SYNC_RATE)
        self.n_evals = kwargs.get('n_evals', 20)
        self.verbose = kwargs.get('verbose', True)
        self.logger = Logger(self.learners, 'step', 'loss', 'reward', 'epsilon', name='dqn')
        self.kwargs = kwargs # store all arguments for printing in __str__

        self.rand_idx = random.randint(0, 100000)
    
    def __str__(self):
        kwargs = self.kwargs
        s  = f"DQN applied to {self.env}\n------------------------------------\n"
        s += f"Number of agents = {len(self.agents)}\n"
        s += f"buffer size = {kwargs.get('buffer_size', 1024)}\n"
        s += f"gamma = {kwargs.get('gamma', 0.99)}\n"
        s += f"learning rate = {kwargs.get('lr', 0.001)}\n"
        s += f"layers = {kwargs.get('layers', [128, 128])}\n"
        s += f"eps_min = {self.eps_min}\n"
        s += f"eps_decay = {self.eps_decay}\n"
        s += f"sync_rate = {self.sync_rate}\n"
        s += f"n_batches = {self.n_batches}\n"
        s += f"sample_size = {self.sample_size}\n"
        s += f"optimizer = {optimizer.__module__.split('.')[-1]}\n"
        s += f"learners = {self.learners}\n"
        s += f'others = {self.others}\n'
        s += f"identifier = {self.rand_idx}  \n"
        s += f"max range = {self.max_distance} \n"
        s += f"learners = {self.learners}"
        s += f"\nfinal reward = {self.eval(self.n_evals)[0]['blue_0']:4.3f}"
        return s

    def run(self, n_iters=10):
        for indx in range(n_iters):
            for _ in range(10):
                episode = self.generate_episode()
                for agent in self.learners:
                    self.buffers[agent].extend(episode[agent][:-1]) # exclude last sample from buffer -> is useless anyway because made after `done`
            if min([len(self.buffers[agent]) for agent in self.learners]) < self.sample_size:
                continue
            losses = {agent: [] for agent in self.learners}
            for _ in range(self.n_batches):
                for agent in self.learners:
                    batch = random.sample(self.buffers[agent], k=self.sample_size)
                    loss = self.agents[agent].update(batch)
                    losses[agent].append(loss)
            if indx % LOG_INTERVAL == 0:
                avg_rwd, std_rwd = self.eval(self.n_evals)
                for agent in self.learners:
                    avg_loss, std_loss = np.mean(losses[agent]), np.std(losses[agent])
                    if self.verbose:
                        print(f"{indx}/{n_iters} - {agent:11s}: loss = {avg_loss:5.4f}, avg reward = {avg_rwd[agent]:5.4f}")
                    self.logger.log(agent, indx, (avg_loss, std_loss), (avg_rwd[agent], std_rwd[agent]), self.epsilon)   

            if indx > 0 and indx % self.sync_rate == 0:
                for agent in self.learners:
                    self.agents[agent].sync()
    
    def eval(self, n):
        rewards = {agent: [] for agent in self.agents}
        for _ in range(n):
            episode = self.generate_episode(train=False)
            for agent in episode:
                rewards[agent].append(np.sum([step.reward for step in episode[agent][:-1]]))
        means, stds = {}, {}
        for agent in self.agents:
            means[agent] = np.mean(rewards[agent]) # TODO: better solution (eg. divide by initial reward)
            stds[agent]  = np.std(rewards[agent]) 

        return means, stds
    
    def generate_episode(self, train=True, render=False):
        self.env.reset()
        episode = {agent: [] for agent in self.env.agents}
        done = False
        for agent in self.env.agent_iter():
            #default_obs = np.zeros(self.env.observation_spaces[agent].shape[0])
            if render:
                self.env.render()
            observation, reward, done, _ =  self.env.last() 
            # set observation, done and reward as next_obs of previous step
            if episode[agent]:
                episode[agent][-1].next_obs = observation['obs']
                episode[agent][-1].next_mask = observation['action_mask']
                episode[agent][-1].reward = reward
                episode[agent][-1].done = done
            action = None if done else self.agents[agent].get_action(observation,
                                                                     epsilon=self.epsilon if train else 0.0)
            self.env.step(action if not done else None)
            episode[agent].append(EpisodeStep(observation['obs'], observation['action_mask'], action,
                                              None, None, None, None))
            if train:
                self.epsilon = max(self.epsilon-self.eps_decay, self.eps_min)
        return episode           


# %%
class Agent:
    """A DQN Agent. Methods:
    * get_action(observation, epsilon): selects action with epsilon-greedy policy based on
                                        current Q estimation (neural Q-net)
    * update(batch): uses deep Q-learning to perform update of Q-net
    """                   
    def __init__(self, name, dims, actions, gamma, lr, layers):
        """initialize class variables

        Args:
            name (str): name of the agent
            dims ([int]): size of observation
            actions ([int]): number of (discrete) actions
            gamma ([float]): discount factor
            lr ([float]): learning rate
            layers ([list]): list of layer sizes for Q-net
        """
        self.n_actions = actions
        self.net = build_network([dims, *layers, actions])
        self.target_net = copy.deepcopy(self.net)
        self.gamma = gamma
        self.optimizer = optimizer(self.net.parameters(), lr=lr)
        self.name = name

    def get_action(self, observation, epsilon=0.0):
        action_mask = observation['action_mask']
        
        if np.random.rand() < epsilon:
            p = action_mask/sum(action_mask)
            action = np.random.choice(range(self.n_actions), p=p)
        else:
            with torch.no_grad():
                qs = self.net(torch.from_numpy(observation['obs']).float())
                qs[action_mask == 0.0] = -np.infty # set all invalid actions to lowest q_value possible
                action = qs.argmax(axis=0).item()
        return action

    def update(self, batch):
        s, mask, a, r, done, next_s, next_mask = zip(*batch)
        qs = self.net(torch.tensor(np.array(s)).float())
        # qs[mask == 0.0] = -np.infty # this doesn't make sense; should be learned, thus ok for target; and broadcast probably wrong
        qs_action = qs[range(len(batch)), a]
        
        with torch.no_grad():
            next_qs = self.target_net(torch.tensor(np.array(next_s)).float())
        next_mask = np.stack(next_mask)
        next_qs[next_mask == 0.0] = -1e10 # set to very low value
        max_next_qs, _ = torch.max(next_qs, dim=-1, keepdim=False)
        
        done = torch.tensor(done, dtype=torch.float32)
        targets = torch.tensor(r) + self.gamma * (1-done) * max_next_qs
        loss = F.mse_loss(qs_action, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def sync(self):
        self.target_net.load_state_dict(self.net.state_dict())
    
    def save(self, rand_idx):
        filename = f"{rand_idx}_{self.name}"
        torch.save(self.net.state_dict(), './nets/'+filename)
    
    def load(self, rand_idx):
        filename = f"{rand_idx}_{self.name}"
        self.net.load_state_dict(torch.load('./nets/'+filename))

# %% Other Agents
class StaticAgent:
    def __init__(self, name):
        self.name = name
    
    def get_action(self, observation, *args, **kwargs):
        return 0

class RandomAgent:
    def __init__(self, name, n_actions):
        self.name = name
        self.n_actions = n_actions
    
    def get_action(self, observation, *args, **kwargs):
        action_mask = observation['action_mask']
        p = action_mask/sum(action_mask)
        action = np.random.choice(range(self.n_actions), p=p)
        return action

# %%

def train(args):
    runner = Runner(episode_length=int(args.length),
                    buffer_size=int(args.buffer_size),
                    n_batches=int(args.n_batches),
                    sample_size=int(args.sample_size), # 64
                    lr=float(args.lr),
                    gamma=float(args.gamma),
                    sync_rate=int(args.sync_rate),
                    layers=[16],
                    eps_steps=int(args.eps_steps),
                    eps_min=0.1,
                    use_mixer=args.use_mixer,
                    n_agents=int(args.n_agents),
                    verbose=True,
                    range=int(args.range),
                    learners=args.learners,
    )
    print(runner)
    runner.run(n_iters=int(args.n_iters))
  
    runner.logger.plot(runner.rand_idx, window=1, text=runner.__str__())
    for agent in runner.learners:
        runner.agents[agent].save(runner.rand_idx)
    print(f'--- Saved with id {runner.rand_idx} ---')
    return runner


def test(id=76872):
    args = Args()
    runner = Runner(episode_length=int(args.length),
                    buffer_size=int(args.buffer_size),
                    n_batches=int(args.n_batches),
                    sample_size=int(args.sample_size), # 64
                    lr=float(args.lr),
                    gamma=float(args.gamma),
                    sync_rate=int(args.sync_rate),
                    layers=[16],
                    eps_steps=int(args.eps_steps),
                    eps_min=0.1,
                    use_mixer=args.use_mixer,
                    n_agents=int(args.n_agents),
                    verbose=True,
                    range=int(args.range),
                    learners=args.learners,
    )
    for agent in runner.learners:
        runner.agents[agent].load(id)
        print('loading agents done')
    print(runner)
    runner.generate_episode(train=False, render=True)

class Args:
        lr = 0.01
        gamma = 0.99
        sync_rate = 50
        n_iters = 500
        buffer_size = 1024
        n_batches = 64
        sample_size = 128
        length = 100
        eps_steps = int(1e5) 
        use_mixer = False
        n_agents = 1
        range=4
        learners='blue'

if __name__ == '__main__':
    args = Args()
    runner = train(args)
    

