from pettingzoo.butterfly import pistonball_v6
import time
import random
import numpy as np 
env = pistonball_v6.env()
env.reset()
while True:
    for agent in env.agent_iter():
        action = random.choice([-1,1])
        env.step((action,))
        env.render()

