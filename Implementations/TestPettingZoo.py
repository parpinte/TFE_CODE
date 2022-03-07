from pettingzoo.magent import battlefield_v3
import time
import random
import numpy as np 
env = battlefield_v3.env()
env.reset()
while True:
    for agent in env.agent_iter():
        action = random.choice([-1,1])
        env.step((action,))
        env.render()

