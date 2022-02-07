""" 
---------- Helper functions ----------
"""
import random
import numpy as np
from matplotlib import pyplot as plt
from torch import nn

def flatten(l):
    "flattens a list"
    return [item for sublist in l for item in sublist]


def build_network(layers):
    """Constructs a sequence of (nn.Linear, nn.ReLU) (except last layer)
    with dimension in `layers`"""
    net = []
    for idx in range(len(layers)-1):
        layer, next_layer = layers[idx], layers[idx+1]
        net.append(nn.Linear(layer, next_layer))
        if idx < len(layers)-2: # NOT the last layer
            net.append(nn.ReLU())
    return nn.Sequential(*net)

def transform_episode(episode):
    """transform dictionary of lists `episode` in list of dictionaries
    with keys = agent names and each containing a single EpisodeStep
    """
    new_episode = []
    agents = list(episode.keys())
    n_steps = len(episode[agents[0]])
    for i in range(n_steps):
        temp = {}
        for agent in agents:
            temp[agent] = episode[agent][i]
        new_episode.append(temp)
    return new_episode

class EpisodeStep:
    def __init__(self, observation, action, reward, done, next_obs, state, next_state):
        self.observation = observation
        self.action = action
        self.reward = reward
        self.done = done
        self.next_obs = next_obs
        self.state = state
        self.next_state = next_state
    
    def __iter__(self):
        all = self.__dict__.values()
        return iter(all)
    
    def __repr__(self):
        s  = f"observation = {self.observation}\n" 
        s += f"action = {self.action}\n"
        s += f"reward = {self.reward}\n"
        s += f"done = {self.done}\n"
        s += f"next_obs = {self.next_obs}\n"
        return s

class Logger:
    def __init__(self, agents, *keywords, name='dqn'):
        "First keyword is used for indexing (should be integer)"
        self.index = keywords[0]
        self.keys = keywords
        self.name = name

        self.content = {}
        for agent in agents:
            self.content[agent] = {key: [] for key in keywords}
    
    def log(self, agent, *values):
        assert len(values) == len(self.keys)
        for key, value in zip(self.keys, values):
            self.content[agent][key].append(value)
    
    def plot(self, id, window=None, text=None):
        if window is None:
            window = np.array([1])
        else:
            window = np.ones(window)/window
        fig, axs = plt.subplots(len(self.keys[1:]), figsize=(12, 12))
        for agent in self.content:
            content = self.content[agent]
            indexes = content[self.index]
            for i, key in enumerate(self.keys[1:]):
                if not isinstance(content[key][0], tuple): # just a single value, no variance
                    axs[i].plot(indexes, np.convolve(content[key], window, mode='same'), label=agent)
                else:
                    vals, stds = zip(*content[key])
                    vals, stds = np.array(vals), np.array(stds)
                    vals_smoothed = np.convolve(vals, window, mode='same')
                    axs[i].plot(indexes, vals_smoothed)
                    axs[i].fill_between(indexes, np.convolve(vals_smoothed-stds, window, mode='same'),
                                        np.convolve(vals_smoothed+stds, window, mode='same'),
                                        alpha=0.2, label=agent)
                axs[i].set_title(key)
                axs[i].legend()
            
        if text is not None:
            plt.subplots_adjust(left=0.25) # make room left of subplots to plot text
            fig.text(0.01, 0.7, text)

        plt.savefig(f'./figures/{self.name}_{id}')
        plt.show()

class Runner:
    def __init__(self, **kwargs) -> None:
        pass

    def log(self, indx, n_iters, cum_loss):
        avg_reward, std_reward = self.eval(self.n_evals)
        if self.verbose:
            print(f"{indx+1}/{n_iters}: avg loss = {cum_loss/self.n_batches:5.4f} | avg reward = {avg_reward:4.3f}")
            self.logger.log('agent_0', indx, cum_loss/self.n_batches, (avg_reward, std_reward), self.epsilon)
            self.writer.add_scalar('avg_reward', avg_reward, indx)
            self.writer.add_scalar('avg_loss', cum_loss/self.n_batches, indx)
            self.writer.add_scalar('epsilon', self.epsilon, indx)

    def eval(self, n):
        episodes = [self.generate_episode(train=False) for _ in range(n)]
        rewards = [np.sum([step['agent_0'].reward for step in episode[:-1]]) for episode in episodes]
        return np.mean(rewards), np.std(rewards)

    def generate_episode(self, render=False, train=True):
        self.env.reset()
        episode = {agent: [] for agent in self.env.agents}
        done = False
        for agent in self.env.agent_iter():
            if render:
                self.env.render()
            observation, reward, done, _ =  self.env.last()
            state = self.env.state()
            # set observation, done and reward as next_obs of previous step
            if episode[agent]:
                episode[agent][-1].next_obs = observation
                episode[agent][-1].next_state = state
                episode[agent][-1].reward = reward
                episode[agent][-1].done = done
            action = self.agents[agent].get_action(observation, epsilon=self.epsilon if train else 0.0)
            self.env.step(action if not done else None)
            episode[agent].append(EpisodeStep(observation, action, None, None, None, state, None))
            if train:
                self.epsilon = max(self.epsilon-self.eps_decay, self.eps_min)
        return transform_episode(episode)
