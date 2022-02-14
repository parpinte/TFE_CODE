## test of collections 
import collections
import random
import numpy as np
Experience = collections.namedtuple('Experience', 
           field_names=['state', 'action', 'reward', 
           'done', 'new_state'])


a = Experience(1,2,3,4,5)
print(a)