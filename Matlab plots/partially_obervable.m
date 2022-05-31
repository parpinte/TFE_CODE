%% Partiallay observable environment 

clear; close all; clc;
width_var = 200;
p = 0.95; 
transparency = 0.1;
factor = 0.5; 
data_1 = csvread('IQL_n_cycle_.csv',2,1);
data_2 = csvread('VDN_partial_observation_ncycles.csv',2,1);

[variance_data_1, mean_data_1] = compute_variation_mean(data_1(:,2), width_var);
[variance_data_2, mean_data_2] = compute_variation_mean(data_2(:,2), width_var);
t = tinv(p, width_var ); 
figure('Name','variation comutation')
hold on 
% xlim([2e5, 3e5]);
semilogy(data_1(:,1), mean_data_1, 'LineWidth',2) 
semilogy(data_1(:,1), mean_data_1 - factor* variance_data_1 , 'r','LineWidth',1)
semilogy(data_1(:,1), mean_data_1 + factor* variance_data_1 , 'r','LineWidth',1)

patch([data_1(:,1) ;flip(data_1(:,1))], [(mean_data_1 - factor* variance_data_1 ); flip(mean_data_1 + factor* variance_data_1 )], ...
                        'r','FaceAlpha',transparency)

semilogy(data_2(:,1), mean_data_2, 'LineWidth',2)
semilogy(data_2(:,1), mean_data_2 - factor.*variance_data_2 , 'b','LineWidth',1)
semilogy(data_2(:,1), mean_data_2 + factor.*variance_data_2 , 'b','LineWidth',1)
patch([data_2(:,1) ;flip(data_2(:,1))], [(mean_data_2 - factor.* variance_data_2 ); flip(mean_data_2 + factor.* variance_data_2 )], ...
                        'b','FaceAlpha',transparency)

grid on
grid minor
xlim([0, 3e5])
xlabel('épisodes','Interpreter','latex')
ylabel('la moyenne de n cycles','Interpreter','latex')
lgd = legend('La moyenne du nombre de cycles pour IQL-OBS-ENV','','','La variation autour de la moyenne IQL-OBS-ENV ','La moyenne du nombre de cycles pour VDN-NO-COM-PARTIAL','','','La variation autour de la moyenne VDN-NO-COM-PARTIAL ','location','southoutside','Interpreter','latex')

title({'Comparaison de la moyenne du nombre de cycles','entre IQL dans un envrironnement totalement' , 'observable et VDN-NO-COM dans un environnement partiellement observable'},'Interpreter','latex')

print('VDN no com vs IQL partial n cycles','-dpng')

start = 666; 
[h,p,ci,stats] = ttest2(data_1(start:end,2),data_2(start:end,2))

%% 
clear; close all; clc;
width_var = 200; % compute the variance each 1000 episodes 
p = 0.95; 
transparency = 0.1; 
factor = 0.5; 
data_1 = csvread('IQL_reward.csv',2,1);
data_2 = csvread('VDN_partial_observation_reward.csv',2,1);

[variance_data_1, mean_data_1] = compute_variation_mean(data_1(:,2), width_var);
[variance_data_2, mean_data_2] = compute_variation_mean(data_2(:,2), width_var);
t = tinv(p, width_var - 1); 
figure('Name','variation comutation')
hold on 
% xlim([2e5, 3e5]);
semilogy(data_1(:,1), mean_data_1, 'LineWidth',2) 
semilogy(data_1(:,1), mean_data_1 - factor.* variance_data_1 , 'r','LineWidth',1)
semilogy(data_1(:,1), mean_data_1 + factor.* variance_data_1 , 'r','LineWidth',1)

patch([data_1(:,1) ;flip(data_1(:,1))], [(mean_data_1 - factor.* variance_data_1 ); flip(mean_data_1 + factor.* variance_data_1 )], ...
                        'r','FaceAlpha',transparency)


semilogy(data_2(:,1), mean_data_2, 'LineWidth',2)
semilogy(data_2(:,1), mean_data_2 - factor.*variance_data_2 , 'b','LineWidth',1)
semilogy(data_2(:,1), mean_data_2 + factor.*variance_data_2 , 'b','LineWidth',1)
patch([data_2(:,1) ;flip(data_2(:,1))], [(mean_data_2 - factor.* variance_data_2 ); flip(mean_data_2 + factor.* variance_data_2 )], ...
                        'b','FaceAlpha',transparency)

grid on
grid minor
xlim([0, 3e5])
xlabel('épisodes','Interpreter','latex')
ylabel('la moyenne de r','Interpreter','latex')
lgd = legend('La moyenne des recompenses pour IQL-OBS-ENV','','','La variation autour de la moyenne IQL-OBS-ENV ','La moyenne des recompenses pour VDN-NO-COM-PARTIAL','','','La variation autour de la moyenne VDN-NO-COM-PARTIAL','location','southoutside','Interpreter','latex')
title({'Comparaison de la moyenne des recompenses','entre IQL dans un envrironnement totalement' , 'observable et VDN-NO-COM dans un environnement partiellement observable'},'Interpreter','latex')
print('VDN no com vs IQL partial recompenses','-dpng')
start = 666; 
[h,p,ci,stats] = ttest2(data_1(start:end,2),data_2(start:end,2))

%% VDN vs VDN with communication in partial observable environment 

clear; close all; clc;
width_var = 200;
p = 0.95; 
transparency = 0.1;
factor = 0.5; 
data_1 = csvread('VDN_partial_observation_continuouscom_n_cycles.csv',2,1);
data_2 = csvread('VDN_partial_observation_ncycles.csv',2,1);

[variance_data_1, mean_data_1] = compute_variation_mean(data_1(:,2), width_var);
[variance_data_2, mean_data_2] = compute_variation_mean(data_2(:,2), width_var);
t = tinv(p, width_var ); 
figure('Name','variation comutation')
hold on 
% xlim([2e5, 3e5]);
semilogy(data_1(:,1), mean_data_1, 'LineWidth',2) 
semilogy(data_1(:,1), mean_data_1 - factor.* variance_data_1 , 'r','LineWidth',1)
semilogy(data_1(:,1), mean_data_1 + factor.* variance_data_1 , 'r','LineWidth',1)

patch([data_1(:,1) ;flip(data_1(:,1))], [(mean_data_1 - factor.* variance_data_1 ); flip(mean_data_1 + factor.* variance_data_1 )], ...
                        'r','FaceAlpha',transparency)

semilogy(data_2(:,1), mean_data_2, 'LineWidth',2)
semilogy(data_2(:,1), mean_data_2 - factor.*variance_data_2 , 'b','LineWidth',1)
semilogy(data_2(:,1), mean_data_2 + factor.*variance_data_2 , 'b','LineWidth',1)
patch([data_2(:,1) ;flip(data_2(:,1))], [(mean_data_2 - factor.* variance_data_2 ); flip(mean_data_2 + factor.* variance_data_2 )], ...
                        'b','FaceAlpha',transparency)

grid on
grid minor
xlim([0, 3e5])
xlabel('épisodes','Interpreter','latex')
ylabel('la moyenne de n cycles','Interpreter','latex')
lgd = legend('La moyenne du nombre de cycles pour VDN-PARTIAL-Qvalues','','','La variation autour de la moyenne VDN-PARTIAL-Qvalues ','La moyenne du nombre de cycles pour VDN-NO-COM-PARTIAL','','','La variation autour de la moyenne VDN-NO-COM-PARTIAL ','location','southoutside','Interpreter','latex')

title({'Comparaison de la moyenne du nombre de cycles','entre VDN-Qvalues et  VDN-NO-COM dans un ','environnement partiellement observable'},'Interpreter','latex')

print('VDN no com vs VDN com partial n cycles','-dpng')

start = 666; 
[h,p,ci,stats] = ttest2(data_1(start:end,2),data_2(start:end,2))


%% 
clear; close all; clc;
width_var = 300; % compute the variance each 1000 episodes 
p = 0.95; 
transparency = 0.1; 
factor = 0.5
data_1 = csvread('VDN_partial_observation_continuouscom_reward.csv',2,1);
data_2 = csvread('VDN_partial_observation_reward.csv',2,1);

[variance_data_1, mean_data_1] = compute_variation_mean(data_1(:,2), width_var);
[variance_data_2, mean_data_2] = compute_variation_mean(data_2(:,2), width_var);
t = tinv(p, width_var - 1); 
figure('Name','variation comutation')
hold on 
% xlim([2e5, 3e5]);
semilogy(data_1(:,1), mean_data_1, 'LineWidth',2) 
semilogy(data_1(:,1), mean_data_1 - factor.* variance_data_1 , 'r','LineWidth',1)
semilogy(data_1(:,1), mean_data_1 + factor* variance_data_1 , 'r','LineWidth',1)

patch([data_1(:,1) ;flip(data_1(:,1))], [(mean_data_1 - factor* variance_data_1 ); flip(mean_data_1 + factor* variance_data_1 )], ...
                        'r','FaceAlpha',transparency)


semilogy(data_2(:,1), mean_data_2, 'LineWidth',2)
semilogy(data_2(:,1), mean_data_2 - factor*variance_data_2, 'b','LineWidth',1)
semilogy(data_2(:,1), mean_data_2 + factor*variance_data_2 , 'b','LineWidth',1)
patch([data_2(:,1) ;flip(data_2(:,1))], [(mean_data_2 - factor* variance_data_2 ); flip(mean_data_2 + factor.* variance_data_2 )], ...
                        'b','FaceAlpha',transparency)

grid on
grid minor
xlim([0, 3e5])
xlabel('épisodes','Interpreter','latex')
ylabel('la moyenne de r','Interpreter','latex')
lgd = legend('La moyenne des recompenses pour VDN-PARTIAL-Qvalues','','','La variation autour de la moyenne VDN-PARTIAL-Qvalues ','La moyenne des recompenses pour VDN-NO-COM-PARTIAL','','','Intervalle de confiance VDN-NO-COM-PARTIAL ','location','southoutside','Interpreter','latex')
title({'Comparaison de la moyenne des recompenses cumulatives','entre VDN-Qvalues et  VDN-NO-COM dans un ','environnement partiellement observable'},'Interpreter','latex')
print('VDN no com vs VDN com partial recompense','-dpng')
start = 666; 
[h,p,ci,stats] = ttest2(data_1(start:end,2),data_2(start:end,2))
