import math 
import numpy as np
import matplotlib.pyplot as plt
import pylab
A = 0.3
B = 0.1
C = 0.1
N_Episodes = 10000
def epsilon(episode):
    standarized_time = (episode - A * N_Episodes)/(B * N_Episodes)
    cosh = np.cosh(math.exp(-standarized_time))
    epsilon = 1.1 - (1 /cosh + (episode * C / N_Episodes))
    return epsilon 

new_time = list(range(0,N_Episodes))
y = [epsilon(episode) for episode in new_time]
plt.plot(new_time,y)
# pylab.show()

parameters_epsilon = {'A': 0.3, 
                'B': 0.1, 
                'C': 0.1}

print(list(parameters_epsilon.values())[0])

class epsilon_params():
    def __init__(self, A = 0.3, B = 0.1, C = 0.1):
        self.A = A
        self.B = B
        self.C = C

parameters_epsilon = epsilon_params(A=0.3, B=0.1, C=0.1)
print(parameters_epsilon.A)