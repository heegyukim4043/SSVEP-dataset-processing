% cd('/Users/gregory/Desktop/analysis/result_figures');
% load classification_results.mat
clear all;

cd('E:\TGI\EEGNET code\result\normal_stimuli_10FCV_230221');

%     to_acc = [];
%     to_f1 = [];
acc_data =[];
for sub = 1:43
    if(~ismember(sub, [22 30]))
        %             if(~ismember(sub, [6 13 15 17 20 38]))
        %                 if(~ismember(sub, [8 11 13 28 31 38]))
        filename = sprintf('result_label_Subject%02d.mat', sub);
        load(filename);
        acc_bar(sub, 1) = mean(acc);
        std_bar(sub,1) = std(acc);
        %                     to_acc = cat(2, to_acc, acc);
        %                     to_f1 = cat(2, to_f1, fval);
        acc_data(sub,:, 1) = acc;
        %                 end
        %             end
    end
end

cd('E:\TGI\EEGNET code\result\normal_hot_230223');

%     to_acc = [];
%     to_f1 = [];
acc_data =[];
for sub = 1:43
    if(~ismember(sub, [22 30]))
        %             if(~ismember(sub, [6 13 15 17 20 38]))
        %                 if(~ismember(sub, [8 11 13 28 31 38]))
        filename = sprintf('result_label_Subject%02d.mat', sub);
        load(filename);
        acc_bar(sub, 2) = mean(acc);
        std_bar(sub,2) = std(acc);
        %                     to_acc = cat(2, to_acc, acc);
        %                     to_f1 = cat(2, to_f1, fval);
        acc_data(sub,:, 2) = acc;
        %                 end
        %             end
    end
end

cd('E:\TGI\EEGNET code\result\normal_cold_230222');

%     to_acc = [];
%     to_f1 = [];
acc_data =[];
for sub = 1:43
    if(~ismember(sub, [22 30]))
        %             if(~ismember(sub, [6 13 15 17 20 38]))
        %                 if(~ismember(sub, [8 11 13 28 31 38]))
        filename = sprintf('result_label_Subject%02d.mat', sub);
        load(filename);
        acc_bar(sub, 3) = mean(acc);
        std_bar(sub,3) = std(acc);
        %                     to_acc = cat(2, to_acc, acc);
        %                     to_f1 = cat(2, to_f1, fval);
        acc_data(sub,:, 3) = acc;
        %                 end
        %             end
    end
end

acc_bar = acc_bar(any(acc_bar,2), :);
std_bar = std_bar(any(std_bar,2), :);

mean_acc_bar = mean(acc_bar, 1);
std_acc_bar = std(acc_bar, 1)./sqrt(43);
group = [1:41];
std_bar = std_bar./sqrt(10);

% for sub = 1:43
%     if acc_bar(sub, 3) <= 0.5
%     acc_bar(sub, :) =0;
%     end
% end

% cd('E:\TGI\EEGNET code\result\result_figures\');
% load TGI_response
%
% for sub =1:43
%     if (~ismember(sub, [22 30]))
% %         if(~ismember(sub, [6 13 15 17 20 22 30 38]))
% %             if(~ismember(sub, [8 11 13 28 31 38]))
%             TGIscore(1, sub) = mean(TGI_Response{sub}(:, TGI_steps{sub} ==1));
%             TGIscore(2, sub) = mean(TGI_Response{sub}(:, TGI_steps{sub} ==9));
%             TGIscore(3, sub) = mean(TGI_Response{sub}(:, TGI_steps{sub} ==2));
%             TGIscore(4, sub) = mean(TGI_Response{sub}(:, TGI_steps{sub} ==8));
%             TGIscore_std(1, sub) = std(TGI_Response{sub}(:, TGI_steps{sub} ==1));
%             TGIscore_std(2, sub) = std(TGI_Response{sub}(:, TGI_steps{sub} ==9));
%             TGIscore_std(3, sub) = std(TGI_Response{sub}(:, TGI_steps{sub} ==2));
%             TGIscore_std(4, sub) = std(TGI_Response{sub}(:, TGI_steps{sub} ==8));
% %             end
% %         end
%     end
%     textposition(sub, 1) = max(acc_bar(sub, 1), acc_bar(sub, 2));
% end
%
% score_err_high = TGIscore + TGIscore_std;
% score_err_low = TGIscore - TGIscore_std;

figure,
bar(group, acc_bar(:,3));
xticks(1:41);
xlabel('Subject');
ylabel('Accuracy');
% yline(0.5, '-', 'LineWidth', 3, 'Color', 'k');
ylim([0.1 1]);
% legend('Cool (Mean acc. 0.62\pm0.02)');
set(gca, 'FontSize', 20);
title('Classification accuracy-Cool stims (Mean acc. 0.65\pm0.01)', 'FontSize', 22);
hold on;
[ngroups, nbars] = size(acc_bar(:,3));
% Calculate the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
% Set the position of each error bar in the centre of the main bar
% Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
yline(0.333, '-', 'LineWidth', 3, 'Color', 'k');
for i = 1:nbars
    % Calculate center of each bar
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, acc_bar(:,3), std_bar(:,3), 'k', 'linestyle', 'none','HandleVisibility', 'on');
end
hold off;
% hold on
% text(1:length(textposition),textposition,num2str(TGIscore(1, :)'),'vert','bottom','horiz','center');
% hold off
%
% for i = 1:2
%     Bi_Class_accavg(:, i) = mean(nonzeros(acc_bar(:, i))); %cool warm
% end

% save('classification_results.mat', 'acc_bar');

% figure,
% scatter(group, acc_data, 'k');
% xlim([0 44]);
% xticks(1:43);
% xlabel('Subject');
% ylabel('Accuracy');
% % yline(0.5, '-', 'LineWidth', 3, 'Color', 'k');
% ylim([0 1]);
% % legend('TGI - Cool', 'TGI - Warm', 'Cool - Warm', 'Random Probability');
% set(gca, 'FontSize', 16);
% title('Classification accuracy', 'FontSize', 22);


%% confusion matrix

% clear all;
for st = 1:3
    test_all_class{st} = [];
    test_all_true{st} = [];
end

cd('E:\TGI\EEGNET code\result\normal_TGI_230227');

for sub = 1:43
    test_class_seq{sub} =[];
    test_seq{sub} =[];
    if(~ismember(sub, [22 30]))
        for tr = 1:10
            filename = sprintf('Subject%d_%d.mat', sub,tr);
            load(filename);
            test_class_seq{sub} = cat(2, test_class_seq{sub}, test_classified_label);
            test_seq{sub} = cat(2, test_seq{sub}, test_true_label_seq);
            test_all_class{1} = cat(2, test_all_class{1}, test_classified_label);
            test_all_true{1} = cat(2, test_all_true{1}, test_true_label_seq);
        end

    end
end

cd('E:\TGI\EEGNET code\result\result_figures\Normal_cold\');

figure,
label = {'20-40','23-37','26-34'};
con=confusionmat(test_all_true{1}, test_all_class{1});
c = confusionchart(con, label,'RowSummary','row-normalized', 'ColumnSummary', 'column-normalized');
title('TGI stimuli (Mean acc. 0.66\pm0.01)');
set(gca, 'FontSize', 18);
filename = sprintf('all_confusionmat.jpg');
saveas(gcf, filename);
close;
%%
cd('E:\TGI\EEGNET code\result\result_figures\Normal_stim_10folds');
for subnum = 1:43
    if (~ismember(subnum, [22 30]))
        figure,
        label = {'20','23-37', '26-34'};
        con=confusionmat(test_seq{subnum}, test_class_seq{subnum});
        c = confusionchart(con, label,'RowSummary','row-normalized', 'ColumnSummary', 'column-normalized');
        c.Title= sprintf('sub = %02d', subnum);
        set(gca, 'FontSize', 14);
        filename = sprintf('confusionmat_subnum_%02d.jpg', subnum);
        saveas(gcf, filename);
        close;
    end
end
%%

figure,
label = {' TGI','Cool'};
subplot(2,1,1);
subnum = 1;
con=confusionmat(test_seq{subnum}, test_class_seq{subnum});
c = confusionchart(con, label ,'RowSummary','row-normalized', 'ColumnSummary', 'column-normalized');
c.Title='mean accuracy = 0.8143';
set(gca, 'FontSize', 14);

subplot(2,1,2);
subnum = 10;
con=confusionmat(test_seq{subnum}, test_class_seq{subnum});
c = confusionchart(con, label ,'RowSummary','row-normalized', 'ColumnSummary', 'column-normalized');
c.Title='mean accuracy = 0.5738';
set(gca, 'FontSize', 14);

%% topoplotting_Wilcoxon
clear all; clc;
cd('E:\TGI\');
load biosemi32_locs
cd('E:\\TGI\EEGNET code\result\result_figures\')

for stim = 1:2
    if stim == 1
        cd('E:\TGI\EEGNET code\result\normal_cold_230222\');
    elseif stim == 2
        cd('E:\TGI\EEGNET code\result\normal_hot_230223\');
    end
    for sub = 1:43
        if(~ismember(sub, [22 30]))
            test_class_seq{sub} =[];
            test_seq{sub} =[];
            disp(sub);
            disp('sub');
            data1 = []; data2 =[]; data3 = [];
            for tr = 1:10
                disp(tr);
                filename = sprintf('Subject%d_%d.mat', sub,tr);
                load(filename);
                x=1:size(attributions_train,1);
                [ep, ch, time] = size(attributions_train);
                data = reshape(attributions_train, [ch, time, ep]);
                data1 = cat(3,data1,data(:,:,ismember(train_true_label(:,1),1)));
                data2 = cat(3,data2,data(:,:,ismember(train_true_label(:,2),1)));
                data3 = cat(3,data3,data(:,:,ismember(train_true_label(:,3),1)));
            end
            data_krus(:,:,1) = mean(data1(:, :,:),3);
            data_krus(:,:,2) = mean(data2(:, :,:),3);
            data_krus(:,:,3) = mean(data3(:, :,:),3);
            for ch = 1:32
                p = kruskalwallis(squeeze(data_krus(ch,:,:)));
                close all;
                p_val(ch,sub, stim) = p;
                if p<=0.05
                    h_val(ch,sub, stim) = 1;
                elseif p>0.05
                    h_val(ch,sub,stim) = 0;
                end
            end
        end
    end
    h_value_top(:, stim) = sum(h_val(:,:, stim), 2);

end
%
%
% figure,
% for st=1:2
%     subplot(1,2,st);
%     topoplot(h_value_top(:,st), biosemi32_locs, 'maplimits', [0 max(max(h_value_top))],...
%         'electrodes', 'ptslabels',  'conv', 'on', 'numcontour', 4);
%     if st ==1
%         title('3');
%     elseif st ==2
%         title('TGI-Warm');
%     elseif st ==3
%         title('Warm-Cool');
%     end
%     set(gca, 'FontSize', 10);
%     colormap(flipud(hot));
% end
%
% figure,
% topoplot(h_value_top(:,st), biosemi32_locs, 'maplimits', [0 max(max(h_value_top))],...
%     'electrodes', 'ptslabels',  'conv', 'on', 'numcontour', 4);
% colormap(flipud(hot));
% c = colorbar;
% % %          c.Label.String = 'relative change';
% title(c, '# significance', 'FontSize', 18);
% c.Position=[0.90, 0.5, 0.03, 0.15];
% c.FontSize = 16;
% % % colorbar;
% % % colormap(hot); caxis([0 0.05]);
%

cd('E:\TGI\EEGNET code\result\result_figures\Normal_stim');
for subj = 1:43
    if(~ismember(subj, [22 30]))
        figure,
        title(subj);
        subplot(1,2,1);
        title('Cool: 3\circC difference');
        topoplot(p_val(:,subj, 1), biosemi32_locs,...
            'electrodes', 'labels','conv', 'on');
        colormap(hot); caxis([0 0.05]);
        set(gca, 'FontSize', 12);
        subplot(1,2,2);
        title('Warm: 3\circC difference');
        topoplot(p_val(:,subj, 2), biosemi32_locs,...
            'electrodes', 'labels','conv', 'on');
        colormap(hot); caxis([0 0.05]);
        subnum = sprintf('Sub = %02d', subj);
        text(-0.95, 0.8, subnum, 'FontSize', 18);
        set(gca, 'FontSize', 12);

        filename = sprintf('wilcoxon_result_subnum_%02d.jpg', subnum);
        saveas(gcf, filename);
        close;
    end
end

%% Score distribution

for sub = 1:43
    if(~ismember(sub, [22 30]))
        for st = 1:3
            if st == 1
                stimuli = [4:6];
            elseif st == 2
                stimuli = [7:9];
            elseif st == 3
                stimuli = [1:3];
            end
            st_distribut(sub, st) = std(TGI_Response{sub}(:, ismember(TGI_steps{sub}, stimuli)));
        end
    end
end

st_distribut = st_distribut(any(st_distribut,2), :);
group = 1:41;

figure,
bar(group, st_distribut(:,1));
xticks(1:41);
xlabel('Subject');
ylabel('Accuracy');
% yline(0.5, '-', 'LineWidth', 3, 'Color', 'k');
% legend('Cool (Mean acc. 0.62\pm0.02)');
set(gca, 'FontSize', 20);
title('standard deviation (score - Cool)', 'FontSize', 22);

%% specific region


% 1-FPAP, 2-F, 3-FC, 4-Cen, 5-CenPar, 6-Pari, 7-PariOcc, 8-Occ, 9-Tem

clear all; clc;

reg{1} = [1 2 29 30];
reg{2} = [3 4 27 28 31]; reg{3} = [5 6 25 26]; reg{4} = [8 23 32]; reg{5} = [9 10 21 22];
reg{6} = [11 12 13 19 20]; reg{7} = [14 18]; reg{8} = [15 16 17]; reg{9} = [7 24];
cd('E:\TGI\');
load biosemi32_locs
cd('E:\\TGI\EEGNET code\result\result_figures\')

for stim = 1:3
    if stim == 1
        cd('E:\TGI\EEGNET code\result\normal_cold_230222\');
    elseif stim == 2
        cd('E:\TGI\EEGNET code\result\normal_hot_230223\');
    elseif stim ==3
        cd('E:\TGI\EEGNET code\result\normal_TGI_230227\')
    end
    for sub = 1:43
        if(~ismember(sub, [22 30]))
            disp(sub);
            disp('sub');
            data1 = []; data2 =[]; data3 = [];
            for tr = 1:10
                disp(tr);
                filename = sprintf('Subject%d_%d.mat', sub,tr);
                load(filename);
                x=1:size(attributions_train,1);
                [ep, ch, time] = size(attributions_train);
                data = reshape(attributions_train, [ch, time, ep]);
                data1 = cat(3,data1,data(:,:,ismember(train_true_label(:,1),1)));
                data2 = cat(3,data2,data(:,:,ismember(train_true_label(:,2),1)));
                data3 = cat(3,data3,data(:,:,ismember(train_true_label(:,3),1)));
            end
            data_krus(:,:,1) = mean(data1(:, :,:),3);
            data_krus(:,:,2) = mean(data2(:, :,:),3);
            data_krus(:,:,3) = mean(data3(:, :,:),3);
            for ch = 1:32
                p = kruskalwallis(squeeze(data_krus(ch,:,:)));
                close all;
                p_val(ch,sub, stim) = p;
                if p<=0.05
                    h_val(ch,sub, stim) = 1;
                elseif p>0.05
                    h_val(ch,sub,stim) = 0;
                end
            end
        end
        h_value_top(:, stim) = sum(h_val(:,:, stim), 2);
    end
end
for subje = 1:43
    for stim =1:3
        for region = 1:9
            h_val_reg_stim(region, subje, stim) = 0;
            if ismember(1, h_val(reg{region}, subje, stim)) ==1
                h_val_reg_stim(region, subje, stim) = 1;
            end
        end
    end
end

all_h_val_reg = sum(h_val_reg_stim, 2);
all_h_val_reg = squeeze(all_h_val_reg);
xaxis = 1:9;

% 1-FPAP, 2-F, 3-FC, 4-Cen, 5-CenPar, 6-Pari, 7-PariOcc, 8-Occ, 9-Tem

figure,
bar(xaxis, all_h_val_reg);
xticks(1:9);
ylim([5 43]);
xticklabels({'FP & AF', 'F', 'FC', 'C', 'CP', 'P', 'PO', 'O', 'T'});
xlabel('Electrode');
ylabel('Number of subjects');
% yline(0.5, '-', 'LineWidth', 3, 'Color', 'k');
legend('Cool', 'Warm', 'TGI');
set(gca, 'FontSize', 20);
title('Common contributed region to classification', 'FontSize', 22);

%% All normal
clear all; clc;

reg{1} = [1 2 29 30];
reg{2} = [3 4 27 28 31]; reg{3} = [5 6 25 26]; reg{4} = [8 23 32]; reg{5} = [9 10 21 22];
reg{6} = [11 12 13 19 20]; reg{7} = [14 18]; reg{8} = [15 16 17]; reg{9} = [7 24];
cd('E:\TGI\');
load biosemi32_locs
cd('E:\TGI\EEGNET code\result\normal_stimuli_10FCV_230221')

for sub = 1:43
    if(~ismember(sub, [22 30]))
        disp(sub);
        disp('sub');
        data1 = []; data2 =[]; data3 = []; data4 = []; data5 =[]; data6 = [];
        for tr = 1:10
            disp(tr);
            filename = sprintf('Subject%d_%d.mat', sub,tr);
            load(filename);
            x=1:size(attributions_train,1);
            [ep, ch, time] = size(attributions_train);
            data = reshape(attributions_train, [ch, time, ep]);
            data1 = cat(3,data1,data(:,:,ismember(train_true_label(:,1),1)));
            data2 = cat(3,data2,data(:,:,ismember(train_true_label(:,2),1)));
            data3 = cat(3,data3,data(:,:,ismember(train_true_label(:,3),1)));
            data4 = cat(3,data4,data(:,:,ismember(train_true_label(:,4),1)));
            data5 = cat(3,data5,data(:,:,ismember(train_true_label(:,5),1)));
            data6 = cat(3,data6,data(:,:,ismember(train_true_label(:,6),1)));
        end
        data_krus(:,:,1) = mean(data1(:, :,:),3);
        data_krus(:,:,2) = mean(data2(:, :,:),3);
        data_krus(:,:,3) = mean(data3(:, :,:),3);
        data_krus(:,:,4) = mean(data4(:, :,:),3);
        data_krus(:,:,5) = mean(data5(:, :,:),3);
        data_krus(:,:,6) = mean(data6(:, :,:),3);

        for ch = 1:32
            p = kruskalwallis(squeeze(data_krus(ch,:,:)));
            close all;
            p_val(ch,sub) = p;
            if p<=0.05
                h_val(ch,sub) = 1;
            elseif p>0.05
                h_val(ch,sub) = 0;
            end
        end
    end
end

for subje = 1:43
    for region = 1:9
        h_val_reg_stim(region, subje) = 0;
        if ismember(1, h_val(reg{region}, subje)) ==1
            h_val_reg_stim(region, subje) = 1;
        end
    end

end

all_h_val_reg = sum(h_val_reg_stim, 2);
all_h_val_reg = squeeze(all_h_val_reg);
xaxis = 1:9;

% 1-FPAP, 2-F, 3-FC, 4-Cen, 5-CenPar, 6-Pari, 7-PariOcc, 8-Occ, 9-Tem

figure,
bar(xaxis, all_h_val_reg, 'k');
xticks(1:9);
ylim([5 43]);
xticklabels({'FP & AF', 'F', 'FC', 'C', 'CP', 'P', 'PO', 'O', 'T'});
xlabel('Electrode');
ylabel('Number of subjects');
% yline(0.5, '-', 'LineWidth', 3, 'Color', 'k');
% legend('Cool', 'Warm', 'TGI');
set(gca, 'FontSize', 20);
title('Common contributed region to classification', 'FontSize', 22);


