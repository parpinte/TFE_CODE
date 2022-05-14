%%% Matlab plots RIAL_2M vs RIAL_Qvales
%% 
clear; close all; clc; 
alpha = 0.98
width = 2.2
rial_2m_loss = csvread("RIAL_2m_blue_down_loss.csv",2,1);
rial_2m_n_cycles = csvread("RIAL_2m_blue_down_n_cycles.csv",2,1);
rial_2m_reward = csvread("RIAL_2m_blue_down_reward.csv",2,1);

rial_qvalues_loss = csvread("RIAL_Qvalues_blue_loss.csv",2,1);
rial_qvalues_n_cycles = csvread("RIAL_Qvalues_blue_down_n_cycles.csv",2,1);
rial_qvalues_reward = csvread("RIAL_Qvalues_blue_down_reward.csv",2,1);
% loss
figure('Name','RIAL_2m vs RIAL Qvalues loss')

rial_2m_loss_smooth = smooth(rial_2m_loss(:,2), alpha);
rial_qvalues_loss_smooth = smooth(rial_qvalues_loss(:,2), alpha);
hold on 
semilogy(rial_2m_loss(:,1),rial_2m_loss(:,2))
semilogy(rial_qvalues_loss(:,1),rial_qvalues_loss(:,2)) 

semilogy(rial_2m_loss(:,1),rial_2m_loss_smooth, 'LineWidth',width)
semilogy(rial_qvalues_loss(:,1),rial_qvalues_loss_smooth, 'LineWidth',width)
grid on 
legend('RIAL-2m-loss','RIAL-Qvalues-loss','RIAL-2m-loss-lisse','RIAL-Qvalues-loss-lisse', 'Interpreter','latex')
xlabel('épisodes','Interpreter','latex')
ylabel('loss','Interpreter','latex')
title('Le loss : RIAL-2m VS RIAL-Qvalues','Interpreter','latex')
print('RIAL-2m VS RIAL-Qvalues loss','-dpng')
% n cycles
figure('Name','RIAL_2m vs RIAL Qvalues n_cycles')
rial_2m_n_cycles_smooth = smooth(rial_2m_n_cycles(:,2), alpha);
rial_qvalues_n_cycles_smooth = smooth(rial_qvalues_n_cycles(:,2), alpha);
hold on 
plot(rial_2m_n_cycles(:,1),rial_2m_n_cycles(:,2))
plot(rial_qvalues_n_cycles(:,1),rial_qvalues_n_cycles(:,2)) 

plot(rial_2m_n_cycles(:,1),rial_2m_n_cycles_smooth, 'LineWidth',width)
plot(rial_qvalues_n_cycles(:,1),rial_qvalues_n_cycles_smooth, 'LineWidth',width)
grid on 
legend('RIAL-2m-n-cycles','RIAL-Qvalues-n-cycles','RIAL-2m-n-cycles-lisse','RIAL-Qvalues-n-cycles-lisse', 'Interpreter','latex')
xlabel('épisodes','Interpreter','latex')
ylabel('Nombre de cycles','Interpreter','latex')
title('Nombre de cycles : RIAL-2m VS RIAL-Qvalues','Interpreter','latex')
print('RIAL-2m VS RIAL-Qvalues n_cycles','-dpng')
%reward 
figure('Name','RIAL_2m vs RIAL Qvalues reward')
rial_2m_reward(:,2) = normalize(rial_2m_reward(:,2)); 
rial_qvalues_reward(:,2) = normalize(rial_qvalues_reward(:,2))
rial_2m_reward_smooth = smooth(rial_2m_reward(:,2), alpha);
rial_qvalues_reward_smooth = smooth(rial_qvalues_reward(:,2), alpha);
hold on 
plot(rial_2m_reward(:,1),rial_2m_reward(:,2))
plot(rial_qvalues_reward(:,1),rial_qvalues_reward(:,2)) 

plot(rial_2m_reward(:,1),rial_2m_reward_smooth, 'LineWidth',width)
plot(rial_qvalues_reward(:,1),rial_qvalues_reward_smooth, 'LineWidth',width)
grid on 
legend('RIAL-2m-récompense','RIAL-Qvalues-récompense','RIAL-2m-récompense-lisse','RIAL-Qvalues-récompense-lisse', 'Interpreter','latex','Location','southeast')
xlabel('épisodes','Interpreter','latex')
ylabel('La Récompense','Interpreter','latex')
title('La Recompense : RIAL-2m VS RIAL-Qvalues','Interpreter','latex')
print('RIAL-2m VS RIAL-Qvalues récompense','-dpng')


%% 

