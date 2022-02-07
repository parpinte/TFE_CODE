import numpy as np
import defense_v0

def test01():
    actions = ['noop', 'left', 'right', 'up','down', 'fire', 'aim0']
    env = defense_v0.env(terrain='central_10x10')
    env.reset()
    #y0 = env.state_.agents['blue_0'].y
    for i in range(1, 4):
        if env.agent_selection == 'blue_0':
            action = actions.index('right')
        else:
            action = 0
        env.step(action)

def test03():
    "fire until out of ammo"
    
    def get_action(agent):
        actions = ['noop', 'left', 'right', 'up','down', 'fire', 'aim0']
        if agent == 'red_0':
            action = 'noop'
        elif agent == 'blue_0':
            if idx == 0:
                action = 'aim0'
            else:
                action = 'fire'
        return actions.index(action)

    env = defense_v0.env(terrain='central_5x5')
    env.reset()
    idx = 0
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        action = get_action(agent) if not done else None
        print(f"{agent=}, {obs}, {action}, {reward}")
        env.step(action)
        env.render()
        idx += 1


def test04():
    "random gameplay"

    def actor(observation):
        mask = observation['action_mask']
        pvals = mask/np.sum(mask)
        action = int(np.random.choice(range(len(pvals)), p=pvals))
        return action

    env = defense_v0.env(terrain='central_5x5')
    env.reset()
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        action = actor(obs) if not done else None
        print(agent, obs, reward, action)
        env.step(action)
        env.render()
        print(env.state())

def test05():
    "test max number of steps"
    def policy(observation, agent):
        mask = observation['action_mask']
        pvals = mask/np.sum(mask)
        action = int(np.random.choice(range(len(pvals)), p=pvals))
        return action
    
    #env = defense_v0.env(terrain='central_5x5', max_cycles=10)
    env = defense_v0.env(terrain='central_7x7_2v2', max_cycles=100)
    env.reset()
    env.render()
    counter = 0
    for agent in env.agent_iter():
        obs, reward, done, info = env.last()
        action = policy(obs, agent) if not done else None
        print(agent, obs, reward, action)
        env.step(action)
        env.render()
        counter += 1
        print(counter, env.state())

def test06():
    "test death of single agent"
    actions = ['noop', 'left', 'right', 'up','down', 'fire', 'aim0']
    steps = {'blue_0': [3, 6, 5, 0],
             'blue_1': [0, 0, 0, 0],
             'red_0':  [0, 0, 0, 0],
             'red_1':  [0, 0, 0, 0]
            }
    
    env = defense_v0.env(terrain='flat_5x5_2v2', max_cycles=100)
    #env = Environment(terrain='flat_5x5_2v2', max_cycles=100)
    env.reset()
    counter = 0
    for agent in env.agent_iter():
        print(counter, agent)
        obs, reward, done, info = env.last()
        action = steps[agent][counter // 4] if not done else None
        #print(agent, obs, reward, action)
        env.step(action)
        #env.render()
        counter += 1
        #print(counter, env.state())
        if counter % 4 == 0:
            print('hold...')

def test07():
    "API test"
    from pettingzoo.test import api_test
    env = defense_v0.env()
    api_test(env, num_cycles=10, verbose_progress=True)



if __name__ == '__main__':
    test04()