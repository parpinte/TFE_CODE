# create graph for epsilon greedy 
import numpy as np
import math
import matplotlib.pyplot as plt


A = 0.5
B = 0.1
C = 0.7
eps = []
N_episodes = 10000

for episode in range(N_episodes):
    standarized_time = (episode - A * N_episodes)/(B * N_episodes)
    cosh = np.cosh(math.exp(-standarized_time))
    epsilon = 1.1 - (1 /cosh + (episode * C / N_episodes))
    eps.append(epsilon)


episodes = list(range(N_episodes))
plt.plot(episodes, eps, label = f"A = {A}, B = {B}, C = {C}")
# seconde plot 
A = 0.3
B = 0.1
C = 0.7
eps = []
N_episodes = 10000

for episode in range(N_episodes):
    standarized_time = (episode - A * N_episodes)/(B * N_episodes)
    cosh = np.cosh(math.exp(-standarized_time))
    epsilon = 1.1 - (1 /cosh + (episode * C / N_episodes))
    eps.append(epsilon)

plt.plot(episodes, eps, label = f"A = {A}, B = {B}, C = {C}")   

#3rd plot  
A = 0.5
B = 0.1
C = 0.1
eps = []
N_episodes = 10000

for episode in range(N_episodes):
    standarized_time = (episode - A * N_episodes)/(B * N_episodes)
    cosh = np.cosh(math.exp(-standarized_time))
    epsilon = 1.1 - (1 /cosh + (episode * C / N_episodes))
    eps.append(epsilon)

plt.plot(episodes, eps, label = f"A = {A}, B = {B}, C = {C}")   

plt.xlabel("Episodes - axis")
plt.ylabel("epsilon - axis")
plt.title("Epsilon Greedy")
plt.legend(loc="upper right")
plt.show()