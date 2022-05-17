%%% example to compute the variance on the reward between VDN and RIAL-2m
clear; close all; clc;
width_var = 200; % compute the variance each 1000 episodes 
p = 0.95; 
transparency = 0.1; 
data_1 = csvread('RIAL_2m_blue_down_reward.csv',2,1);
data_2 = csvread('IQL_reward.csv',2,1);

[variance_data_1, mean_data_1] = compute_variation_mean(data_1(:,2), width_var);
[variance_data_2, mean_data_2] = compute_variation_mean(data_2(:,2), width_var);
t = tinv(p, width_var - 1); 
figure('Name','variation comutation')
hold on 
semilogy(data_1(:,1), mean_data_1, 'LineWidth',2) 
semilogy(data_1(:,1), mean_data_1 - t.* variance_data_1 / sqrt(width_var ), 'r','LineWidth',1)
semilogy(data_1(:,1), mean_data_1 + t.* variance_data_1 / sqrt(width_var), 'r','LineWidth',1)

patch([data_1(:,1) ;flip(data_1(:,1))], [(mean_data_1 - t.* variance_data_1 / sqrt(width_var )); flip(mean_data_1 + t.* variance_data_1 / sqrt(width_var ))], ...
                        'r','FaceAlpha',transparency)


semilogy(data_2(:,1), mean_data_2, 'LineWidth',2)
semilogy(data_2(:,1), mean_data_2 - t.*variance_data_2 / sqrt(width_var ), 'b','LineWidth',1)
semilogy(data_2(:,1), mean_data_2 + t.*variance_data_2 / sqrt(width_var ), 'b','LineWidth',1)
patch([data_2(:,1) ;flip(data_2(:,1))], [(mean_data_2 - t.* variance_data_2 / sqrt(width_var)); flip(mean_data_2 + t.* variance_data_2 / sqrt(width_var))], ...
                        'b','FaceAlpha',transparency)
grid on
grid minor

xlabel('épisodes','Interpreter','latex')
ylabel('la moyenne de r','Interpreter','latex')
title('Comparaison de la moyenne entre RIAL-2m et IQL','Interpreter','latex')
legend('La moyenne des récompenses RIAL-2m','','','Intervalle de confiance RIAL-2m 95\%','La moyenne des récompenses pour IQL','','','Intervalle de confiance IQL 95\%','location','southeast','Interpreter','latex')
print('IQL vs RIAL-2m variances','-dpng')


%% RIAL-2m vs RIAL-10m
clear; close all; clc;
width_var = 200; % compute the variance each 1000 episodes 
p = 0.95; 
transparency = 0.1; 
data_1 = csvread('RIAL_2m_blue_down_reward.csv',2,1);
data_2 = csvread('RIAL_10m_blue_up_reward.csv',2,1);

[variance_data_1, mean_data_1] = compute_variation_mean(data_1(:,2), width_var);
[variance_data_2, mean_data_2] = compute_variation_mean(data_2(:,2), width_var);
t = tinv(p, width_var - 1); 
figure('Name','variation comutation')
hold on 
semilogy(data_1(:,1), mean_data_1, 'LineWidth',2) 
semilogy(data_1(:,1), mean_data_1 - t.* variance_data_1 / sqrt(width_var ), 'r','LineWidth',1)
semilogy(data_1(:,1), mean_data_1 + t.* variance_data_1 / sqrt(width_var), 'r','LineWidth',1)

patch([data_1(:,1) ;flip(data_1(:,1))], [(mean_data_1 - t.* variance_data_1 / sqrt(width_var )); flip(mean_data_1 + t.* variance_data_1 / sqrt(width_var ))], ...
                        'r','FaceAlpha',transparency)


semilogy(data_2(:,1), mean_data_2, 'LineWidth',2)
semilogy(data_2(:,1), mean_data_2 - t.*variance_data_2 / sqrt(width_var ), 'b','LineWidth',1)
semilogy(data_2(:,1), mean_data_2 + t.*variance_data_2 / sqrt(width_var ), 'b','LineWidth',1)
patch([data_2(:,1) ;flip(data_2(:,1))], [(mean_data_2 - t.* variance_data_2 / sqrt(width_var)); flip(mean_data_2 + t.* variance_data_2 / sqrt(width_var))], ...
                        'b','FaceAlpha',transparency)
grid on
grid minor

xlabel('épisodes','Interpreter','latex')
ylabel('la moyenne de r-cumulative','Interpreter','latex')
title({'Comparaison de la moyenne de la',' recompense entre RIAL-2m et RIAL-10m'},'Interpreter','latex')
legend('La moyenne des récompenses RIAL-2m','','','Intervalle de confiance RIAL-2m 95\%','La moyenne des récompenses pour RIAL-10m','','','Intervalle de confiance RIAL-10m 95\%','location','southeast','Interpreter','latex')

print('RIAm-10m vs RIAL-2m moyenne r','-dpng')
% test statistique 
start = 630;
data_1_sample = data_1(start:end, 2);
data_2_sample = data_2(start:end, 2);
[h,p,ci,stats] = ttest2(data_1_sample, data_2_sample)
% 
data_1 = csvread('RIAL_2m_blue_down_n_cycles.csv',2,1);
data_2 = csvread('RIAL_10m_blue_up_n_cycles.csv',2,1);

[variance_data_1, mean_data_1] = compute_variation_mean(data_1(:,2), width_var);
[variance_data_2, mean_data_2] = compute_variation_mean(data_2(:,2), width_var);
t = tinv(p, width_var - 1); 
figure('Name','variation comutation')
hold on 
semilogy(data_1(:,1), mean_data_1, 'LineWidth',2) 
semilogy(data_1(:,1), mean_data_1 - t.* variance_data_1 / sqrt(width_var ), 'r','LineWidth',1)
semilogy(data_1(:,1), mean_data_1 + t.* variance_data_1 / sqrt(width_var), 'r','LineWidth',1)

patch([data_1(:,1) ;flip(data_1(:,1))], [(mean_data_1 - t.* variance_data_1 / sqrt(width_var )); flip(mean_data_1 + t.* variance_data_1 / sqrt(width_var ))], ...
                        'r','FaceAlpha',transparency)


semilogy(data_2(:,1), mean_data_2, 'LineWidth',2)
semilogy(data_2(:,1), mean_data_2 - t.*variance_data_2 / sqrt(width_var ), 'b','LineWidth',1)
semilogy(data_2(:,1), mean_data_2 + t.*variance_data_2 / sqrt(width_var ), 'b','LineWidth',1)
patch([data_2(:,1) ;flip(data_2(:,1))], [(mean_data_2 - t.* variance_data_2 / sqrt(width_var)); flip(mean_data_2 + t.* variance_data_2 / sqrt(width_var))], ...
                        'b','FaceAlpha',transparency)
grid on
grid minor
xlim([1.5e5 3e5])
xlabel('épisodes','Interpreter','latex')
ylabel('la moyenne de r-cumulative','Interpreter','latex')
title({'Comparaison de la moyenne de la',' recompense entre RIAL-2m et RIAL-10m'},'Interpreter','latex')
legend('La moyenne des récompenses RIAL-2m','','','Intervalle de confiance RIAL-2m 95\%','La moyenne des récompenses pour RIAL-10m','','','Intervalle de confiance RIAL-10m 95\%','location','northeast','Interpreter','latex')

start = 666;
data_1_sample = data_1(start:end, 2);
data_2_sample = data_2(start:end, 2);
[h,p,ci,stats] = ttest2(data_1_sample, data_2_sample)

print('RIAm-10m vs RIAL-2m n cycles','-dpng')
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
%% variance RIAl-2m vs IQL

clear; close all; clc; 
width_var = 200; % compute the variance each 1000 episodes
start_mean = 800; 
data_1 = csvread('RIAL_2m_blue_down_reward.csv',2,1);
data_2 = csvread('IQL_reward.csv',2,1);

variance_data_1 = compute_variation(data_1(:,2), width_var);
variance_data_2 = compute_variation(data_2(:,2), width_var);




figure('Name','variation computation')
semilogy(data_1(:,1), variance_data_1, 'LineWidth',2)
hold on 
semilogy(data_2(:,1), variance_data_2, 'LineWidth',2)

grid on

xlabel('épisodes','Interpreter','latex')
ylabel('la variance','Interpreter','latex')
title({'Comparaison de la variance de la recompense',' cumulative entre RIAL-2m et IQL'},'Interpreter','latex')
legend('La variance pour RIAL-2m','La variance pour IQL','Interpreter','latex','Location','southwest')
 print('IQL vs RIAL-2m variances','-dpng')
% kolmogorov smirnov test for testing the normality of the datas 

mean_1 = mean(data_1(start_mean:end,2)) * ones(length(data_1)- start_mean + 1,1);
mean_2 = mean(data_2(start_mean:end,2)) * ones(length(data_2)- start_mean + 1,1);
close all 
figure('Name','la récompeses pour IQL et RIAL')
hold on 
grid on 
plot(data_1(start_mean:end ,1),data_1(start_mean:end ,2))
plot(data_2(start_mean:end ,1),data_2(start_mean:end ,2))
plot(data_1(start_mean:end ,1),mean_1 , 'r','LineWidth',2.2)

plot(data_2(start_mean:end ,1),mean_2 , 'LineWidth',2.2)
legend('La récompense RIAL-2m','La récompense RIAL-2m','La moyenne de R pour RIAL-2m','La moyenne de R pour RIAL-2m','Interpreter','latex','Location','southwest')
xlabel('épisodes','Interpreter','latex')
ylabel('La récompense','Interpreter','latex')
title({'Comparaison de la recompense',' cumulative entre RIAL-2m et IQL'},'Interpreter','latex')
close all
%print('IQL vs RIAL-2m mean recompense','-dpng')  
[h,p,ci,stats] = ttest2(data_1(start_mean:end,2),data_2(start_mean:end,2))






