function [done] = make_plots(filename_1, filename_2, lim, leg, labelx, labely, tit, printname, save)
    data_1 = load(filename_1,'cumulative_reward_blue');
    data_2 = load(filename_2,'cumulative_reward_blue');
    data_1 = data_1.cumulative_reward_blue; 
    data_2 = data_2.cumulative_reward_blue;
    episodes = 1:length(data_1);
    episodes_2 = 1:length(data_2);
    mean_1 = mean(data_1)*ones(length(data_1),1);
    mean_2 = mean(data_2)*ones(length(data_1),1);
    
    figure('Name','reward');
    hold on 
    plot(episodes, data_1);
    
    plot(episodes_2, data_2);
    plot(episodes, mean_1,'--b','LineWidth',2)
    plot(episodes_2, mean_2,'--r','LineWidth',2)
    xlim(lim);
    legend(leg, 'Interpreter','latex','Location','south');
    a = xlabel(labelx,'Interpreter','latex'); a.FontSize = 15; 
    b = ylabel(labely,'Interpreter','latex'); b.FontSize = 15; 
    title(tit, 'Interpreter','latex','FontSize',15); 
    if save == true
        print(printname, '-dpng')
    end
    % taux de réussite 
    ratio_1 = sum(data_1 > 0)/length(data_1)
    ratio_2 = sum(data_2 > 0)/length(data_2)
    % test statistique 
    [h,p,ci,stats] = ttest2(data_1,data_2)
    % Anova test 
    %prepare the data 
    table = [data_1',data_2'];
    [p,tbl,stats] = anova1(table);

end