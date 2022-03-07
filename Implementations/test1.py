
from pettingzoo.mpe import simple_spread_v2
import time 

env = simple_spread_v2.env(N = 3)
env.reset()
env.render() 

for agent in env.agent_iter(): 
    