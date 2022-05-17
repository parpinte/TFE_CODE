%% IQL curves 

clear; close all; clc; 

alpha = 0.98;
width = 2.5;

curve1_A = csvread("IQL_loss.csv",2,1);
curve2_A = csvread("IQL_n_cycle_.csv",2,1);
curve3_A = csvread("IQL_reward.csv",2,1);

curve1_B = csvread("RIAL_2m_blue_down_loss.csv",2,1);
curve2_B = csvread("RIAL_2m_blue_down_n_cycles.csv",2,1);
curve3_B = csvread("RIAL_2m_blue_down_reward.csv",2,1);




% loss
figure('Name','IQL loss')

curve1_A_smooth = smooth(curve1_A(:,2), alpha);
curve1_B_smooth = smooth(curve1_B(:,2), alpha);
hold on 
semilogy(curve1_A(:,1),curve1_A(:,2))

semilogy(curve1_B(:,1),curve1_B(:,2))
semilogy(curve1_A(:,1),curve1_A_smooth, 'LineWidth',width)
semilogy(curve1_B(:,1),curve1_B_smooth, 'LineWidth',width)

grid on 
legend('IQL-loss','RIAL-2m-loss','IQL-loss-lisse','RIAL-2m-loss-lisse', 'Interpreter','latex')
xlabel('épisodes','Interpreter','latex')
ylabel('loss','Interpreter','latex')
title('Le loss : IQL vs RIAL-2m','Interpreter','latex')
% print('IQL loss','-dpng')

% n cycles
figure('Name','IQL n cycles and reward ')
subplot(1,2,1)
curve2_A_smooth = smooth(curve2_A(:,2), alpha);
curve2_B_smooth = smooth(curve2_B(:,2), alpha);

hold on 
plot(curve2_A(:,1),curve2_A(:,2))
plot(curve2_B(:,1),curve2_B(:,2))


plot(curve2_A(:,1),curve2_A_smooth, 'LineWidth',width)
plot(curve2_B(:,1),curve2_B_smooth, 'LineWidth',width)

grid on 
legend('IQL-n-cycles','RIAL-2m-ncycles','IQL-n-cycles-lisse','RIAL-2m-ncycles-lisse', 'Interpreter','latex')
xlabel('épisodes','Interpreter','latex')
ylabel('n-cycles','Interpreter','latex')
title('fig-a : Le nombre de cycles : IQL vs RIAL-2m ','Interpreter','latex')


% reward 
subplot(1,2,2)

curve3_A_smooth = smooth(curve3_A(:,2), alpha);
curve3_B_smooth = smooth(curve3_B(:,2), alpha);

hold on 
plot(curve3_A(:,1),curve3_A(:,2))
plot(curve3_B(:,1),curve3_B(:,2))


plot(curve3_A(:,1),curve3_A_smooth, 'LineWidth',width)
plot(curve3_B(:,1),curve3_B_smooth, 'LineWidth',width)

grid on 
legend('IQL-récompense','RIAL-2m-récompense','IQL-récompense-lisse','RIAL-2m-récompense-lisse', 'Interpreter','latex','Location','southeast')
xlabel('épisodes','Interpreter','latex')
ylabel('récompense','Interpreter','latex')
title('fig-b : La recompense : IQL vs RIAL-2m ','Interpreter','latex')

print('IQL vs RIAL-2m ncycles and reward','-dpng')




%%