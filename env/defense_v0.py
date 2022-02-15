"""
Actions:    0 -> noop
            1 -> left
            2 -> right
            3 -> up
            4 -> down
            5 -> fire
            6 -> aim0
            7 -> aim1
"""

# TODO: write suite of unittests
# TODO: verify firing not allowed when blocked

from itertools import product
import functools

import numpy as np
from matplotlib import pyplot as plt
plt.ion()
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

from gym import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers

RANGE = 4 # 10 #4
AMMO  = 5
STEP = -0.01 # reward for making a step

## --------------------------------------------------------------------------------------------------------------------------
def env(terrain="flat_5x5", max_cycles=100, max_distance=RANGE):
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = Environment(terrain, max_cycles, max_distance)
    env = wrappers.CaptureStdoutWrapper(env)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env
## --------------------------------------------------------------------------------------------------------------------------

def distance(agent, other):
    x1, y1  = agent.x, agent.y
    x2, y2 = other.x, other.y
    return np.sqrt((x2-x1)**2  + (y2-y1)**2)

class State:
    def __init__(self, agents, obstacles) -> None:
        self.agents = agents
        self.obstacles = obstacles
    
    @property
    def occupied(self):
        "returns list of occupied squares"
        squares = self.obstacles[:]
        for agent in self.agents.values():
            squares.append((agent.x, agent.y))
        return squares
    
    def get_observation(self, agent): # TODO: improve (include ID)
        # TODO: add visibility information?
        if isinstance(agent, str):
            agent = self.agents[agent]
        observation = { 'self': agent.to_array(),
                        'team': [other.to_array() for other in self.agents.values()
                                    if other.team==agent.team and other != agent],
                        'others': [other.to_array() for other in self.agents.values()
                                    if other.team!=agent.team],
                        'obstacles': self.obstacles
                    }
        # squash all values in a single array
        observation = np.concatenate([np.squeeze(val).flatten() for val in observation.values()])

        return observation

    def to_array(self):
        """returns state in array-form
        Structure:  * obstacles: x, y
                    * agents:   * x, y
                                * alive (0 or 1)
                                * ammo 
                                * aim
        """
        if len(self.obstacles) > 0:
            obstacle_arr = np.concatenate([np.array([x, y]) for x, y in self.obstacles])
        else:
            obstacle_arr = np.array([])
        agent_arr = np.concatenate([agent.to_array() for agent in self.agents.values()])
        arr = np.concatenate([
            obstacle_arr,
            agent_arr]
        )
        return arr
    
    @staticmethod
    def from_array(arr):
        obstacle_arr = arr[:-20].reshape(-1, 2)
        obstacles = [tuple(row) for row in obstacle_arr]

        agent_arr = arr[-20:].reshape(-1, 5)
        #agents = {team: [] for team in ['blue', 'red']}
        agents = []
        ii = 0
        for team in ['blue', 'red']:
            for id in [0, 1]:
                agents.append(Agent.from_array(id, team, agent_arr[ii, :]))
                ii += 1
        agents = {agent.name : agent for agent in agents}
        return State(agents, obstacles)

    def winner(self):
        if all([not agent.alive for agent in self.agents.values() if agent.team == 'blue']): # all agents of team 'blue' are dead
            return 'red'
        elif all([not agent.alive for agent in self.agents.values() if agent.team == 'red']):
            return 'blue'
        else:
            return None

    def reward(self,agent):
        winner = self.winner()
        if winner is None:
            reward = 0
        elif winner == agent.team:
            reward = 1
        else:
            reward = -1
        return reward
    
    def get_other_agent(self, agent, id=None):
        if id is None:
            id = agent.aim.id
        other_team = 'blue' if agent.team == 'red' else 'red'
        other_agent = self.agents[other_team + '_' + str(id)]
        return other_agent

class Agent:
    def __init__(self, id, team, x, y) -> None:
        self.name = f"{team}_{id}" 
        self.id = id
        self.team = team # "blue" or "red"
        self.name = f"{self.team}_{self.id}"
        self.set_attributes(x, y, True, 5, -1)
    
    def set_attributes(self, x, y, alive, ammo, aim):
        self.alive = True
        self.ammo =  AMMO
        self.aim  = -1
        self.set_position(x, y)
    
    def set_position(self, x, y):
        self.x, self.y = x, y
    
    def to_array(self):
        aim = self.aim.id if self.aim != -1 else -1
        return np.array([self.x, self.y, int(self.alive), self.ammo, aim])
    
    @staticmethod
    def from_array(id, team, arr):
        x, y = arr[0], arr[1]
        alive = arr[2] == True
        aim = int(arr[3])
        ammo = int(arr[4])
        agent = Agent(id, team, x, y)
        agent.set_attributes(x, y, alive, ammo, aim)
        return agent

class Environment(AECEnv):
    metadata = {'render.modes': ['human'], "name": "defense_v0"}

    def __init__(self, terrain, max_cycles, max_distance) -> None:
        self.terrain = load_terrain(terrain)
        self.max_cycles = max_cycles
        self.obstacles = self.terrain['obstacles']
        self.size = self.terrain['size']
        self.vmap = self.get_vis_map()
        self.range = max_distance

        # define agents
        n_agents = len(self.terrain['blue']) # assumes equal number of agents on each side
        agents = [Agent(id, 'blue', *pos) for id, pos in enumerate(self.terrain['blue'])] + \
                 [Agent(id, 'red',  *pos) for id, pos in enumerate(self.terrain['red'])]
        self.agents_ = {agent.name : agent for agent in agents}
        self.state_ = State(self.agents_, self.obstacles)
        self.possible_agents = list(self.agents_.keys())
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        actions = ['noop', 'left', 'right', 'up','down', 'fire']
        actions.extend([f'aim{id}' for id in range(n_agents)])
        self.actions = {i: action for i, action in enumerate(actions)}
        self.inv_actions = {action: i for i, action in enumerate(actions)}
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        self._obs_dim = self.state_.get_observation('blue_0').shape[0] # for flexibility
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        return spaces.Dict({'obs': spaces.Box(low=-1, high=max(self.size, AMMO), shape=(self._obs_dim,), dtype=float),
                            'action_mask': spaces.Box(low=0, high=1, shape=(len(self.actions),), dtype=float)
                            })

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(len(self.actions))

    def state(self):
        return self.state_.to_array()
    
    def __str__(self):
        s = '_' * (self.size + 2) + '\n'
        for x in range(self.size):
            s += '|'
            for y in range(self.size):
                if (x, y) in self.obstacles:
                    s += 'x'
                else:
                    s += '.'
            s += '|\n'
        s += '-' * (self.size + 2)
        return s
    
    def is_visible(self, p1, p2, N=100):
        x1, y1 = p1
        x2, y2 = p2
        xs = np.linspace(x1, x2, N)
        ys = np.linspace(y1, y2, N)
        for x, y in zip(xs, ys):
            if (round(x), round(y)) in self.obstacles:
                return False 
        return True 
    
    def get_vis_map(self):
        "creates visibility map"
        vis_map = {}
        for x1 in range(self.size):
            for y1 in range(self.size):
                for x2 in range(self.size):
                    for y2 in range(self.size):
                        vis_map[((x1, y1), (x2, y2))] = self.is_visible((x1, y1), (x2, y2))
        return vis_map
    
    def allowed(self, agent):
        "returns mask of allowed actions for agent"
        # TODO: might also be method of State
        if isinstance(agent, str):
            agent = self.agents_[agent]
        mask = np.zeros(len(self.actions))
        if not agent.alive: # no actions allowed
            return mask
        occupied = self.state_.occupied
        for action in self.actions.values():
            if action == 'noop': # always allowed
                mask[self.inv_actions[action]] = 1. 
            elif action == 'left':
                if agent.y > 0 and (agent.x, agent.y-1) not in occupied: # conditions
                    mask[self.inv_actions[action]] = 1.
            elif action == 'right':
                if agent.y < self.size-1 and (agent.x, agent.y+1) not in occupied:
                    mask[self.inv_actions[action]] = 1.
            elif action == 'up':
                if agent.x > 0 and (agent.x-1, agent.y) not in occupied:
                    mask[self.inv_actions[action]] = 1.
            elif action == 'down':
                if agent.x < self.size-1 and (agent.x+1, agent.y) not in occupied:
                    mask[self.inv_actions[action]] = 1.
            elif action == 'aim0' or action == 'aim1':
                mask[self.inv_actions[action]] = 1. # always allowed
            elif action == 'fire':
                if agent.aim != -1 and agent.ammo > 0:  # checks aiming and ammo
                    # checks line-of-sight between agents - TODO: should this be given to or learned by agent?
                    if self.vmap[((agent.x, agent.y), (agent.aim.x, agent.aim.y))]: 
                        mask[self.inv_actions[action]] = 1
        return mask
    
    def observe_(self, agent):
        action_mask = self.allowed(agent)
        obs = self.state_.get_observation(agent)
        return {'obs': obs, 'action_mask': action_mask}
    
    def observe(self, agent):
        return self.observations[agent]
    
    def reset(self):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        # reset agents and state
        agents = [Agent(id, 'blue', *pos) for id, pos in enumerate(self.terrain['blue'])] + \
                 [Agent(id, 'red',  *pos) for id, pos in enumerate(self.terrain['red'])]
        self.agents_ = {agent.name : agent for agent in agents}
        self.state_ = State(self.agents_, self.obstacles)

        # reset the AEC parameters
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        #self.state = {agent: self.state_ for agent in self.agents} # clashes with state() method
        self.observations = {agent: self.observe_(agent).copy() for agent in self.agents}
        self.steps = 0
        
        # Our agent_selector utility allows easy cyclic stepping through the agents list.
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        '''
            step(action) takes in an action for the current agent (specified by
            agent_selection) and needs to update
            - rewards
            - _cumulative_rewards (accumulating the rewards)
            - dones
            - infos
            - agent_selection (to the next agent)
            And any internal state used by observe() or render()
        '''
        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent

            # below is a hack to jump to correct next agent
            current_agent = self.agent_selection
            next_agent = self.agents[(self.agents.index(current_agent) + 1) % len(self.agents)]
            self._was_done_step(action)
            self.agent_selection = next_agent
            return


        agent = self.agents_[self.agent_selection] # select Agent object
        self._cumulative_rewards[self.agent_selection] = 0 

        mask = self.allowed(agent)
        assert mask[action] == 1., f"action {action} ('{self.actions[action]}') not allowed"

        self.infos = {agent: {'status': 'succes'} for agent in self.agents}

        if self.actions[action] == 'left':
            agent.y -= 1
        elif self.actions[action] == 'right':
            agent.y += 1
        elif self.actions[action] == 'up':
            agent.x -= 1
        elif self.actions[action] == 'down':
            agent.x += 1
        elif self.actions[action][:3] == 'aim':
            other = int(self.actions[action][3])
            agent.aim = self.state_.get_other_agent(agent, other)
        elif self.actions[action] == 'fire':
            agent.ammo -= 1
            # determine distance to other agent 
            other_agent = self.state_.get_other_agent(agent)

            # firing cant work if not visible => is now handled by disallowing firing when no line-of-sight
            # if not self.vmap[((agent.x, agent.y), (other_agent.x, other_agent.y))]: 
            #    self.infos[agent] = {'status': 'fail'}
            
            # firing only works if in range
            if distance(agent, other_agent) <= self.range:
                other_agent.alive = False
        
        ## AEC part:
        # TODO: quid self._clear_rewards()? This only works when no intermediate rewards
        winner = self.state_.winner()
        if winner is not None:
            for agent in self.agents:
                if self.agents_[agent].team == winner:    # agent's team has won
                    self.rewards[agent] = 1
                    self.dones[agent] = True
                    self.infos[agent]['winner'] = 'self'
                else:                                       # # agent's team has lost
                    self.rewards[agent] = -1
                    self.dones[agent] = True
                    self.infos[agent]['winner'] = 'other'
        else:
            self._cumulative_rewards[self.agent_selection] += STEP

        for agent in self.agents:
            # additional done criterium: if dead, agent is done
            if not self.agents_[agent].alive:
                self.dones[agent] = True
        
        
        # avoid collisons - this is a makeshift solution, effectively giving priority
        # for certain moves to agents that come earlier in the cycle
        for agent in self.agents:
            self.observations[agent]['action_mask'] = self.observe_(agent)['action_mask']

        # To be performed only after complete cycle over all agents:
        # observe the current state for all agents
        if self._agent_selector.is_last():
            self.observations = {agent: self.observe_(agent).copy() for agent in self.agents}
            
            # check if max_cycles hasn't been exceeded; if so, set all agents to done
            self.steps += 1
            if self.steps >= self.max_cycles:
                for a in self.agents:
                    self.dones[a] = True
        else:
            # no rewards are allocated until both players give an action
            #self._clear_rewards()
            pass # TODO: check this

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    def _make_graph(self):
        _, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.axis('equal')
        
        self.ax.set_xlim([0, self.size])
        self.ax.set_xticks(range(0, self.size))
        self.ax.set_ylim([0, self.size])
        self.ax.set_yticks(range(0, self.size))
        self.ax.grid()

        self.patches = {}
        for x, y in self.obstacles:
            self.ax.add_patch(Rectangle((y+0.05, self.size-x+0.05-1), width=.9, height=.9, color='black'))
        for agent in self.agents_.values():
            self.patches[agent.name] = Circle((agent.y+0.5, self.size-agent.x+0.5-1), radius=0.45, color=agent.team,
                                               alpha=1 if agent.alive else .3)
            # self.patches[agent.name] =Rectangle((agent.y+0.05, self.size-agent.x+0.05-1), width=0.9, height=0.9,
            #                                       color=agent.team, alpha=1 if agent.alive else .3))
            self.ax.add_patch(self.patches[agent.name])
        
        self.lines = {}
        for agent1, agent2 in product(self.agents_.values(), self.agents_.values()):
            self.lines[(agent1.name, agent2.name)] = Line2D([agent1.y+.5, agent2.y+.5],
                                                            [self.size-agent1.x-0.5, self.size-agent2.x-0.5],
                                                            alpha=0., color=agent1.team, linewidth=0.5)
            self.ax.add_line(self.lines[(agent1.name, agent2.name)])
    
    def render(self, mode='human'):
        if not hasattr(self, 'ax'):
            self._make_graph()
        for agent in self.agents_.values():
            self.patches[agent.name].set(center=(agent.y+0.5, self.size-agent.x+0.5-1),
                                         alpha=1 if agent.alive else .3)
            self.ax.text(agent.y+.45, self.size-agent.x-0.6, str(agent.id), fontsize=12, color='white')

        for agent1, agent2 in product(self.agents_.values(), self.agents_.values()):
            self.lines[(agent1.name, agent2.name)].set_xdata((agent1.y+.5, agent2.y+.5))
            self.lines[(agent1.name, agent2.name)].set_ydata((self.size-agent1.x-0.5, self.size-agent2.x-0.5))
            if agent1.aim == agent2:
                self.lines[(agent1.name, agent2.name)].set_alpha(1.)
            else:
                self.lines[(agent1.name, agent2.name)].set_alpha(0.)

        plt.show()
        plt.pause(1.0)

## --utilities -------------------------------------------------

def write_terrain(name, terrain):
    """Writes representation of terrain to file with
    filename 'name.ter'

    Args:
        name ([str]): name of terrain
        terrain ([list]): terrain object as list; elements are coordinates of objects
    """
    size = terrain['size']
    s = render_terrain(terrain)
    with open(f'env/terrains/{name}_{size}x{size}.ter', 'w') as f:
        f.write(s)

def load_terrain(name):
    terrain = {'obstacles': [], 'blue': [], 'red': []}
    path = 'env/terrains/'
    with open(path + name + '.ter', 'r') as f:
    #with open('terrains/' + name + '.ter', 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0:
                terrain['size'] = len(line)-1
            for key, ident in [('obstacles', 'x'), ('blue', '1'), ('red', '2')]:
                for n in [n for n in range(len(line)) if line.find(ident, n) == n]: # all occurences of 'x'
                    terrain[key].append((idx, n))
    return terrain

def make_terrain(size):
    terrain = {'size': size,
                'obstacles': []}
    if isinstance(size, int):
        s0, s1 = size, size
    elif isinstance(size, tuple):
        s0, s1 = size
    terrain['obstacles'].append((s0//2-1, s1//2-1))
    terrain['obstacles'].append((s0//2,   s1//2-1))
    terrain['obstacles'].append((s0//2-1, s1//2))
    terrain['obstacles'].append((s0//2,   s1//2))
    return terrain

def render_terrain(terrain):
    size = terrain['size']
    obstacles = terrain['obstacles']
    s = ''
    for x in range(size):
        for y in range(size):
            if (x, y) in obstacles:
                s += 'x'
            else:
                s += '.'
        s += '\n'
    return s

    
if __name__ == '__main__':
    pass