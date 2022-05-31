%% RIAL-2m vs RIAL-10m
clear; close all; clc; 
filename_1 = 'RIAL-10m.mat';
filename_2 = 'RIAL-2m.mat';
lim = [0, 200];
leg = {'la récompense cumulative RIAL-10m','La récompense cumulative RIAL-2m',...
            'La moyenne pour RIAL-10m','La moyenne pour RIAL-2m'};
labelx = 'épisodes';
labely = 'La récompense cumulative';
title = {"Comparaison entre RIAL-2m et RIAL-10m ","sur base de la recompense cumulative"}; 
printname = "stat RIAl-2m vs RIAL-10m"; 
save = true; 
make_plots(filename_1, filename_2, lim, leg, labelx, labely, title, printname, save)
xticklabels({'RIAL-10m','RIAL-2m'})
b = ylabel('Récompense cumulative','Interpreter','latex'); b.FontSize = 15; 
print(strcat(printname,'anova'),'-dpng')
figure(3)
print(strcat(printname,'box'),'-dpng')
%% RIAL 10 m vs Qvalues 
clear; close all; clc; 
filename_1 = 'RIAL-10m.mat';
filename_2 = 'RIAL_Qvalues.mat';
lim = [0, 200];
leg = {'la récompense cumulative RIAL-10m','La récompense cumulative RIAL-Qvalues',...
            'La moyenne pour RIAL-10m','La moyenne pour RIAL-Qvalues'};
labelx = 'épisodes';
labely = 'La récompense cumulative';
title = {"Comparaison entre RIAL-Qvalues  "," et RIAL-10m sur base de la recompense cumulative"}; 
printname = "stat RIAL-Qvalues vs RIAL-10m"; 
save = true; 
make_plots(filename_1, filename_2, lim, leg, labelx, labely, title, printname, save)
xticklabels({'RIAL-10m','RIAL_Qvalues'})
b = ylabel('Récompense cumulative','Interpreter','latex');b.FontSize =15; 
print(strcat(printname,'anova'),'-dpng')
figure(3)
print(strcat(printname,'box'),'-dpng')
%% RIAL Qvalues vs RIAL Action 
clear; close all; clc; 
filename_1 = 'RIAL_Qvalues.mat';
filename_2 = 'RIAL-Action.mat';
lim = [0, 200];
leg = {'la récompense cumulative RIAL-Qvalues','La récompense cumulative RIAL-Action',...
            'La moyenne pour RIAL-Qvalues','La moyenne pour RIAL-Action'};
labelx = 'épisodes';
labely = 'La récompense cumulative';
title = {"Comparaison entre RIAL-Action  "," et RIAL-Qvalues sur base de la recompense cumulative"}; 
printname = "stat RIAL-Qvalues vs RIAL-Action"; 
save = true; 
make_plots(filename_1, filename_2, lim, leg, labelx, labely, title, printname, save)
xticklabels({'RIAL_Qvalues','RIAL_Action'})
b = ylabel('Récompense cumulative','Interpreter','latex'); b.FontSize = 15; 
print(strcat(printname,'anova'),'-dpng')
figure(3)
print(strcat(printname,'box'),'-dpng')
%% VDN no com vs RIALQvalues
clear; close all; clc; 
filename_1 = 'RIAL_Qvalues.mat';
filename_2 = 'VDN_nocom.mat';
lim = [0, 200];
leg = {'la récompense cumulative RIAL-Qvalues','La récompense cumulative VDN-NO-COM',...
            'La moyenne pour RIAL-Qvalues','La moyenne pour VDN-NO-COM'};
labelx = 'épisodes';
labely = 'La récompense cumulative';
title = {"Comparaison entre VDN-NO-COM  "," et RIAL-Qvalues sur base de la recompense cumulative"}; 
printname = "stat RIAL-Qvalues vs VDN-NO-COM"; 
save = true; 
make_plots(filename_1, filename_2, lim, leg, labelx, labely, title, printname, save)
xticklabels({'RIAL_Qvalues','VDN-NO-COM'})
b = ylabel('Récompense cumulative','Interpreter','latex'); b.FontSize = 15
print(strcat(printname,'anova'),'-dpng')
figure(3)
print(strcat(printname,'box'),'-dpng')

%% VDN no com vs VDN Qvalues 
clear; close all; clc; 
filename_1 = 'VDN_nocom.mat';
filename_2 = 'VDN_Qvalues.mat';
lim = [0, 300];
leg = {'la récompense cumulative VDN-NO-COM','La récompense cumulative VDN-Qvalues',...
            'La moyenne pour VDN-NO-COM','La moyenne pour VDN-Qvalues'};
labelx = 'épisodes';
labely = 'La récompense cumulative';
title = {"Comparaison entre VDN-Qvalues  "," et VDN-NO-COM sur base de la recompense cumulative"}; 
printname = "stat VDN-Qvalues vs VDN-NO-COM"; 
save = true; 
make_plots(filename_1, filename_2, lim, leg, labelx, labely, title, printname, save)
xticklabels({'VDN-NO-COM','VDN-Qvalues'})
b = ylabel('Récompense cumulative','Interpreter','latex'); b.FontSize  =15; 
print(strcat(printname,'anova'),'-dpng')
figure(3)
print(strcat(printname,'box'),'-dpng')
%% partial : VDN vs VDN with communication
clear; close all; clc; 
filename_1 = 'VDN_partial_observation.mat';
filename_2 = 'VDN_partial_observation_Qvalues.mat';
lim = [0, 200];
leg = {'la récompense cumulative VDN-partial','La récompense cumulative VDN-partial-Qvalues',...
            'La moyenne pour VDN-partial','La moyenne pour VDN-partial-Qvalues'};
labelx = 'épisodes';
labely = 'La récompense cumulative';
title = {"Comparaison entre VDN-partial  "," et VDN-partial-Qvalues sur base de la recompense cumulative"}; 
printname = "stat VDN-partial vs VDN-partial-Qvalues"; 
save = true; 
make_plots(filename_1, filename_2, lim, leg, labelx, labely, title, printname, save)
xticklabels({'VDN-partial','VDN-partial-Qvalues'})
b = ylabel('Récompense cumulative','Interpreter','latex'); b.FontSize = 15; 
print(strcat(printname,'anova'),'-dpng')
figure(3)
print(strcat(printname,'box'),'-dpng')