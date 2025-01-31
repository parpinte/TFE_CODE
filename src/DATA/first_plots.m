%% 
clear; close all; clc; 
weight = 0.98; 
epsilon = csvread("Sim_5m_7x7_central_eps.csv",2,2);
episodes = csvread("Sim_5m_7x7_central_eps.csv",2);
episodes = episodes(:,2);
loss = csvread("Sim_5m_7x7_central_loss.csv",2,2);
n_cycles = csvread("Sim_5m_7x7_central_Ncycles.csv",2,2);
reward = csvread("Sim_5m_7x7_central_reward.csv",2,2);


figure; 
subplot(1,2,1)
hold on 
semilogy(episodes, epsilon);
% semilogy(episodes, smooth(epsilon, weight), 'LineWidth',3)
grid on 
grid minor
x = xlabel('épisodes','Interpreter','latex'); 
x.FontSize = 12
y = ylabel('epsilon','Interpreter','latex');
y.FontSize = 15
title('Epsilon en fonction du temps','Interpreter','latex')


subplot(1,2,2)
hold on 
semilogy(episodes, loss);
semilogy(episodes, smooth(loss, weight), 'LineWidth',3)
grid on 
grid minor
x = xlabel('épisodes','Interpreter','latex'); 
x.FontSize = 12
y = ylabel('loss','Interpreter','latex');
y.FontSize = 15
legend('Le loss','Le loss-lisse')
title('Le loss en fontion du temps','Interpreter','latex')
print('lossandeps1','-dpng')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure; 
subplot(1,2,1)
hold on 
semilogy(episodes, n_cycles);
semilogy(episodes, smooth(n_cycles, weight), 'LineWidth',3)
grid on 
grid minor
x = xlabel('épisodes','Interpreter','latex'); 
x.FontSize = 12
y = ylabel('epsilon','Interpreter','latex');
y.FontSize = 15
legend('n-cycles','n-cycles-lisse')
title('Epsilon en fonction du temps','Interpreter','latex')


subplot(1,2,2)
hold on 
semilogy(episodes, reward);
semilogy(episodes, smooth(reward, weight), 'LineWidth',3)
grid on 
grid minor
x = xlabel('épisodes','Interpreter','latex'); 
x.FontSize = 12
y = ylabel('loss','Interpreter','latex');
y.FontSize = 15
legend('recompenses','recompenses-lisse')
title('Le loss en fontion du temps','Interpreter','latex')
print('ncycles_reward','-dpng')

