import gym
import numpy as np
import time


# import the envronment 
env = gym.make("CartPole-v1")
env.reset()

print(f"env.observation_space.high = {env.observation_space.high}")
print(f"env.observation_space.low = {env.observation_space.low}")


'''
The environment gives continuous values so we need to discetize the different given observations
'''
# create the q table accordings to the bins
def create_q_table(nbBins, env):
    HighState = env.observation_space.high
    LowState = env.observation_space.low

    bins = [ 
    np.linspace(LowState[0], HighState[0],nbBins),
    np.linspace(-4, 4, nbBins),
    np.linspace(LowState[2], HighState[2], nbBins),
    np.linspace(-4, 4, nbBins)
            ]

    # create q table 

    q_table = np.random.uniform(low = -1, high = 0, size = ([nbBins] * len(bins) + [env.action_space.n]))
    return bins, q_table

# function to create the discrete state 

def get_discrete_state(state, bins):
    stateIndex = []
    for i in range(4):
        stateIndex.append(np.digitize(state[i],bins[i]) -1)
    # have to transform this list into a tuple to can use it as index 
    return tuple(stateIndex)


# define the different parameters of the problem 
nbEpisodes = 20 # number of episodes  
epsilon = 1 # eps greedy (for exploration)( not a fixed value)
alpha = 0.2 # learning rate 
gamma = 0.8 # dscount factor 
showTime = 10000 # every how many episode we ganna show the simulation 
epsilon_step_decaying = epsilon / (nbEpisodes - 50)
# 


# begin simulation 
bins, q_table = create_q_table(nbBins = 10,env = env)

for episode in range(nbEpisodes):
    discrete_state = get_discrete_state(env.reset(), bins)

    done = False

    if episode % showTime == 0 & episode > 10000:
        render = True 
        print(f"episode n° = {episode}")
    else:
        render = False 

    while not done:
        if np.random.random() > epsilon: 
            action = np.argmax(q_table[discrete_state])
        else: 
            action = env.action_space.sample()
        
        if render:
            env.render()
            
        new_state, reward, done, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state, bins)
        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state + (action, )]

        # update the new q 
        new_q = (1 - alpha) * current_q + alpha *(reward + gamma * max_future_q)
        # put the new value for each position 
        q_table[discrete_state + (action, )] = new_q

        discrete_state = new_discrete_state

        # perform epsilon decayin 

    epsilon = epsilon - epsilon_step_decaying





# test part 


discrete_state = get_discrete_state(env.reset(), bins)
action = np.argmax(q_table[discrete_state])
new_state, reward, done, _ = env.step(action)
while not done:
    new_discrete_state = get_discrete_state(new_state, bins)
    action = np.argmax(q_table[discrete_state])
    discrete_state = get_discrete_state(env.reset(), bins)
    env.render()
    time.sleep(0.1)
    
    
    
    
