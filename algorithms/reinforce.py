"""Independet REINFORCE for DEFENSE (with action mask)
We implement the simplest policy gradient method: REINFORCE.

A policy gradient algorithm immediately optimizes the policy itself
(we don't need the notion of a value-action function) by computing
the gradient of the policy (via the policy-gradient theorem) and 
performing gradient ascent.

WITH parameter (weight) sharing => same policy network for each team

Williams, R.J. Simple statistical gradient-following algorithms
for connectionist reinforcement learning. Mach Learn 8, 229–256 
(1992). https://doi.org/10.1007/BF00992696
"""

from typing import Dict
import random

import numpy as np 
import torch
from torch import optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from utilities import build_network, EpisodeStep

PRINT_PERIOD = 10

# hack to allow import of env
import sys; sys.path.insert(0, '.')
from env import defense_v0 

class Agent:
    """A Reinforce Agent"""
    def __init__(self, name, net, gamma) -> None:
        self.name = name
        self.policy = net
        self.gamma = gamma
    
    def get_action(self, observation):
        action_mask = observation['action_mask']
        with torch.no_grad():
            logits = self.policy(torch.from_numpy(observation['obs']).float())
            logits[action_mask == 0.0] = -1e6 # set logits of not-allowed actions to very low value
            action = Categorical(logits=logits).sample()
        return action.item()

    def update(self, batch, rtgs):
        """See Sutton - 13.3 REINFORCE: Monte Carlo Policy Gradient
        + apply logarithm trick"""
        obs, masks, actions, rewards, dones, _, _ = zip(*batch[:-1]) # skip last elem of batch (contains no useful information)
        rtgs = torch.from_numpy(np.stack(rtgs)).float()  
        logits = self.policy(torch.from_numpy(np.stack(obs)).float()) # result of policy network are `logits`
        log_actions = logits[range(len(actions)), actions] # select only the values for the actions taken
        loss = -torch.mean(log_actions * rtgs) # a minus because we want gradient ascent !

        self.policy.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

        return loss.item()
    
class RandomAgent:
    def __init__(self, name, n_actions):
        self.name = name
        self.n_actions = n_actions
    
    def get_action(self, observation, *args, **kwargs):
        action_mask = observation['action_mask']
        p = action_mask/sum(action_mask)
        action = np.random.choice(range(self.n_actions), p=p)
        return action

class Runner:
    """A Runner class.
    * __init__: initializes environment and agents
    * generate_episode: generates a single episode
    * run: generates episodes and updates each each based on episode results
    * rewards_to_go: compute, for each agent, the rtgs based on (one or more) episodes
    * eval: evaluates current policies
    * log: logs results to tensorboard
    """
    def __init__(self, terrain='flat_5x5', lr=0.05) -> None:
        self.env = defense_v0.env(terrain=terrain, max_cycles=20)
        self.env.reset()
        self.gamma = 0.99

        nets = {}
        for team in ['blue', 'red']:
            nets[team] = build_network([self.env.observation_space(team+'_0')['obs'].shape[0], 128, 
                                        self.env.action_space(team+'_0').n])
            nets[team].optimizer = optim.Adam(params=nets[team].parameters(), lr=lr)

        # select the agent that are going to learn - in general: all (self.env.agents)
        self.learners = self.env.agents.copy()
        self.learners = ['blue_0']

        self.agents = {}
        for agent in self.env.agents:
            team = agent[:-2] # 'blue' or 'red'
            if agent in self.learners:
                self.agents[agent] = Agent(name=agent, net=nets[team], gamma=self.gamma)
            else:
                self.agents[agent] = RandomAgent(name=agent, n_actions=self.env.action_space(team+'_0').n)
        
        
        self.rand_idx = random.randint(0, 100000)
        self.writers = {agent: SummaryWriter(log_dir=f"runs/reinforce_{str(self.rand_idx)}_{str(agent)}") for agent in self.learners}
    
    def __str__(self):
        s = f"Runner with id {str(self.rand_idx)}"
        return s

    def run(self, n_iters=10):
        for indx in range(n_iters):
            episode = self.generate_episode()
            rtgs = self.rewards_to_go(episode)
            losses = {agent: [] for agent in self.agents}
            for name in self.learners:
                agent = self.agents[name]
                loss = agent.update(episode[name], rtgs[name])
                losses[name].append(loss)
            
            evals = self.eval()
            self.log(indx, losses, evals)

    def eval(self) -> Dict:
        results = {agent: {} for agent in self.learners}
        for agent in self.learners:
            episodes = [self.generate_episode()[agent] for _ in range(10)]
            results[agent]['reward'] = np.mean([np.sum([step.reward for step in episode[:-1]]) for episode in episodes])
            results[agent]['length'] = np.mean([len(episode) for episode in episodes])
        return results
    
    def log(self, indx, losses, results) -> None:
        for agent in self.learners:
            avg_loss, std_loss = np.mean(losses[agent]), np.std(losses[agent])
            if indx % PRINT_PERIOD == 0:
                print(f"{indx} - {agent:11s}: loss = {avg_loss:5.4f}, reward = {results[agent]['reward']:5.4f}, length = {results[agent]['length']:3.2f}")
            self.writers[agent].add_scalar('loss', avg_loss, indx)
            self.writers[agent].add_scalar('reward', results[agent]['reward'], indx)
            self.writers[agent].add_scalar('length', results[agent]['length'], indx)

    def generate_episode(self):
        self.env.reset()
        episode = {agent: [] for agent in self.env.agents}
        done = False
        for agent in self.env.agent_iter():
            observation, reward, done, _ =  self.env.last() 
            # set observation, done and reward as next_obs of previous step
            if episode[agent]:
                episode[agent][-1].next_obs = observation['obs']
                episode[agent][-1].next_mask = observation['action_mask']
                episode[agent][-1].reward = reward
                episode[agent][-1].done = done
            
            action = None if done else self.agents[agent].get_action(observation)
            
            self.env.step(action)
            episode[agent].append(EpisodeStep(observation['obs'], observation['action_mask'], action,
                                              None, None, None, None))
        return episode
     
    def rewards_to_go(self, batch):
        rtgs = {agent : [] for agent in batch}
        for agent in batch:
            R = 0
            for step in reversed(batch[agent][:-1]):
                if step.done:
                    R = 0
                R = self.gamma*R +  step.reward
                rtgs[agent].insert(0, R)
        return rtgs


def train():
    pass

if __name__ == '__main__':
    runner = Runner(terrain='flat_5x5', lr=0.00005)
    print(runner)
    runner.run(n_iters=5000)