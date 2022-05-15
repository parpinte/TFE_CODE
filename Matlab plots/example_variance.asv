%%% example to compute the variance on the reward between VDN and RIAL-2m
clear; close all; clc;
width_var = 200; % compute the variance each 1000 episodes 
data_1 = csvread('RIAL_2m_blue_down_reward.csv',2,1);
data_2 = csvread('VDN_Qvalues_reward.csv',2,1);

variance_data_1 = compute_variation(data_1(:,2), width_var);
variance_data_2 = compute_variation(data_2(:,2), width_var);

figure('Name','variation comutation')
semilogy(data_1(:,1), variance_data_1, 'LineWidth',2)
hold on 
semilogy(data_2(:,1), variance_data_2, 'LineWidth',2)
grid on

xlabel('épisodes','Interpreter','latex')
ylabel('la variance','Interpreter','latex')
title('Comparaison de la variance entre RIAL-2m et VDN-Qvalues','Interpreter','latex')
legend('La variance pour RIAL-2m','La variance pour VDN-Qvalues','Interpreter','latex')



%% RIAL-2m vs RIAL-10m
clear; close all; clc; 
width_var = 200; % compute the variance each 1000 episodes 
data_1 = csvread('RIAL_2m_blue_down_reward.csv',2,1);
data_2 = csvread('RIAL_10m_blue_up_reward.csv',2,1);

variance_data_1 = compute_variation(data_1(:,2), width_var);
variance_data_2 = compute_variation(data_2(:,2), width_var);

figure('Name','variation comutation')
semilogy(data_1(:,1), variance_data_1, 'LineWidth',2)
hold on 
semilogy(data_2(:,1), variance_data_2, 'LineWidth',2)
grid on

xlabel('épisodes','Interpreter','latex')
ylabel('la variance','Interpreter','latex')
title({'Comparaison de la variance de la récompense',' cumulative entre RIAL-2m et RIAL-2m'},'Interpreter','latex')
legend('La variance pour RIAL-2m','La variance pour RIAL-2m','Interpreter','latex')
%% RIAL-10m vs VDN

clear; close all; clc; 
width_var = 200; % compute the variance each 1000 episodes 
data_1 = csvread('VDN_Qvalues_reward.csv',2,1);
data_2 = csvread('RIAL_10m_blue_up_reward.csv',2,1);

variance_data_1 = compute_variation(data_1(:,2), width_var);
variance_data_2 = compute_variation(data_2(:,2), width_var);

figure('Name','variation comutation')
semilogy(data_1(:,1), variance_data_1, 'LineWidth',2)
hold on 
semilogy(data_2(:,1), variance_data_2, 'LineWidth',2)
grid on

xlabel('épisodes','Interpreter','latex')
ylabel('la variance','Interpreter','latex')
title({'Comparaison de la variance de la récompense',' cumulative entre RIAL-2m et RIAL-2m'},'Interpreter','latex')
legend('La variance pour RIAL-2m','La variance pour RIAL-2m','Interpreter','latex')


%% RIAL Qvalues vs VDN 

clear; close all; clc; 
width_var = 200; % compute the variance each 1000 episodes 
data_1 = csvread('VDN_Qvalues_reward.csv',2,1);
data_2 = csvread('RIAL_10m_blue_up_reward.csv',2,1);

variance_data_1 = compute_variation(data_1(:,2), width_var);
variance_data_2 = compute_variation(data_2(:,2), width_var);

figure('Name','variation comutation')
semilogy(data_1(:,1), variance_data_1, 'LineWidth',2)
hold on 
semilogy(data_2(:,1), variance_data_2, 'LineWidth',2)
grid on

xlabel('épisodes','Interpreter','latex')
ylabel('la variance','Interpreter','latex')
title({'Comparaison de la variance de la récompense',' cumulative entre RIAL-2m et RIAL-2m'},'Interpreter','latex')
legend('La variance pour RIAL-2m','La variance pour RIAL-2m','Interpreter','latex')


