# Data Analysis 
import pandas as pd
import matplotlib.pyplot as plt
from torch import fbgemm_linear_fp16_weight_fp32_activation

file_path_5m_central = "./DATA/Sim_5m_7x7_central_eps.csv"
file_path_5m_central_loss = "./DATA/Sim_5m_7x7_central_loss.csv"
file_path_5m_central_ncyles = "./DATA/Sim_5m_7x7_central_Ncycles.csv"
file_path_5m_central_reward = "./DATA/Sim_5m_7x7_central_reward.csv"

fig1, (ax1, ax2) = plt.subplots(1,2)
fig1.suptitle("Epsilon et le Loss en fonction du temps")

eps = pd.read_csv(file_path_5m_central)
loss = pd.read_csv(file_path_5m_central_loss)
smooth_factor = 0.99
smooth_loss = loss.ewm(alpha=(1 - smooth_factor)).mean()
ax1.plot(eps['Step'], eps['Value'], label="epsilon")
ax1.grid(which='minor')
ax1.set_xlabel('épisodes')
ax1.set_ylabel('epsilon')
ax1.legend(loc='upper right')
ax2.plot(loss['Step'], loss['Value'], alpha= 0.4, label = 'Loss')
ax2.plot(smooth_loss['Step'], smooth_loss['Value'], Label = 'La moyenne du Loss')
ax2.set_yscale('log')
ax2.grid(which='minor')
ax2.set_ylim([0.25, 1])
ax2.set_xlabel('episodes')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper right')
#plt.yscale('log')
#plt.grid(which='minor') # , color='#EEEEEE', linestyle=':', linewidth=0.0005)
ax1.minorticks_on()
ax2.minorticks_on()



fig2 , (ax3 ,ax4) = plt.subplots(1,2)
fig2.suptitle('Le nombre de cycles et La récompense en fonction du temps')
ncycle = pd.read_csv(file_path_5m_central_ncyles)
reward = pd.read_csv(file_path_5m_central_reward) 
smooth_ncycle = ncycle.ewm(alpha=(1 - smooth_factor)).mean()
smooth_reward = reward.ewm(alpha=(1 - smooth_factor)).mean()

ax3.plot(ncycle['Step'], ncycle['Value'], label = 'N cycles', alpha = 0.4)
ax3.plot(smooth_ncycle['Step'], smooth_ncycle['Value'], label = 'La moyenne de N cycles')
ax3.grid(which='minor')
ax3.set_xlabel('épisodes')
ax3.set_ylabel('Nombre de cycles')

ax4.plot(reward['Step'], reward['Value'], label = 'Récompence', alpha = 0.4)
ax4.plot(smooth_reward['Step'], smooth_reward['Value'], label = 'La moyenne des récompenses')
ax4.grid(which='minor')
ax4.set_xlabel('épisodes')
ax4.set_ylabel('Nombre de cycles')
ax3.legend(loc='upper right')
ax4.legend(loc='upper right')
ax3.minorticks_on()
ax4.minorticks_on()


plt.show()
