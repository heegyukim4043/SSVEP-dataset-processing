%% init toolbox
addpath('eeglab2022.0');
addpath('fieldtrip-20190618');
eeglab; 
ft_defaults;

clc;clear;close all;

%% init param.

clear;
path_dataset= 'SSVEP_data';
save_path = 'results';

% input_param.method = 'stdCCA';
% input_param.method = 'FBCCA';
% input_param.method = 'ITCCA';
% input_param.method = 'TRCA';
input_param.method = 'CITCCA';
% 'stdCCA', 'FBCCA', 'CITCCA', 'TRCA'

%%% common param.
% channel selection
% harmonics

% input_param.ch_list ={'OZ','O1','O2'};
input_param.ch_list  = {'O1','O2','OZ','POZ','PO3','PO4','PO5','PO6','PO7','PO8','PZ'};
input_param.reref_ch = {'M1','M2'};
input_param.harmonics = 3; % in ref
input_param.window_len = [0 1000]; % ms


switch input_param.method
    case 'stdCCA'
        
        input_param.filter = [13, 22*input_param.harmonics+1] ;
        
    case 'FBCCA'
        
        for har_idx =  1: input_param.harmonics
            input_param.filter_bank(har_idx,:) = [14*har_idx-1, 22*input_param.harmonics+1]; 
        end
        input_param.weight_ab = [1, 0];
        
    case 'ITCCA'
        
        input_param.fold = 5;  % fold 6 -> train: test = 5: 1, fold 5 -> train: test = 4: 2
        input_param.filter = [13 22*input_param.harmonics+1];
        
    case 'TRCA'
        
        input_param.fold = 5;
        input_param.filter = [13 22*input_param.harmonics+1];
        
    case 'CITCCA'
        
        input_param.fold = 5;
        input_param.filter = [13 22*input_param.harmonics+1];
end


%% preproc & classification

subj_idx = 1 ;

clear eeg;
load(sprintf('%s/raw_eeg_ssvep_subj_%02d.mat',path_dataset,subj_idx));


% generate ref(Y) and EEG(X)
switch (input_param.method)
    
    case 'stdCCA'
        time = input_param.window_len(1)/1000:1/eeg.srate:input_param.window_len(end)/1000;
        %time = floor(time);
        time(end) = [];
        
        for har_idx = 1: input_param.harmonics
            for class_idx = 1 : length(eeg.freq)
              Y([1:2]+2*(har_idx-1),:,class_idx) = ...
                  [sin(2*har_idx*eeg.freq(class_idx)*pi*time);...
                  cos(2*har_idx*eeg.freq(class_idx)*pi*time)];
            end
        end
        
        
    case 'FBCCA'
        time = input_param.window_len(1)/1000:1/eeg.srate:input_param.window_len(end)/1000;
       % time = floor(time);
        time(end) = [];
        
        for har_idx = 1: input_param.harmonics
            for class_idx = 1 : length(eeg.freq)
              Y([1:2]+2*(har_idx-1),:,class_idx) = ...
                  [sin(2*har_idx*eeg.freq(class_idx)*pi*time);...
                  cos(2*har_idx*eeg.freq(class_idx)*pi*time)];
            end
        end
        
    case 'ITCCA'
%         time = input_param.window_len(1)/1000:1/eeg.srate:input_param.window_len(end)/1000;
%         time(end) = [];
        
        train_list=nchoosek(1:size(eeg.data,4),input_param.fold-1);
        for i = 1 : size(train_list,1)
           temp_test_list = 1:size(eeg.data,4);
           temp_test_list(train_list(i,:)) = [];
           test_list(i,:) = temp_test_list;
        end
        
        for class_idx = 1 : length(eeg.freq)
            for fold_idx = 1 : size(train_list,1)
                template_data=squeeze(eeg.data(:,:,class_idx,train_list(fold_idx,:)));
                
                template_data = reref(template_data,find(ismember(eeg.chan_locs,input_param.reref_ch)),'keepref','on');
                for trial_idx = 1 : size(template_data,3)
                    template_data(:,:,trial_idx) = ...
                        ft_preproc_bandpassfilter(template_data(:,:,trial_idx),eeg.srate,input_param.filter,4,'but');
                end
                
                if (eeg.window(1)<= input_param.window_len(1) & eeg.window(end) >= input_param.window_len(end))
                    
                    window_size = [input_param.window_len(1)/1000*eeg.srate:input_param.window_len(end)/1000*eeg.srate]-eeg.srate*eeg.window(1)/1000;
                    window_size = floor(window_size);
                    window_size(1) = [];
                    Y_IT(:,:,class_idx,fold_idx) = mean(template_data(ismember(eeg.chan_locs,input_param.ch_list),...
                        window_size,...
                        :),3);
                end
                
            end
        end
        
        
    case 'TRCA'
        train_list = nchoosek(1:size(eeg.data,4),input_param.fold-1);
        for i  = 1 : size(train_list,1)
            temp_test_list = 1 : size(eeg.data,4);
            temp_test_list(train_list(i,:)) = [];
            test_list(i,:) = temp_test_list;
        end
        
         
        for class_idx = 1 : length(eeg.freq)
            for fold_idx = 1 : size(train_list,1)
                template_data=squeeze(eeg.data(:,:,class_idx,train_list(fold_idx,:)));
                
                template_data = reref(template_data,find(ismember(eeg.chan_locs,input_param.reref_ch)),'keepref','on');
                for trial_idx = 1 : size(template_data,3)
                    template_data(:,:,trial_idx) = ...
                        ft_preproc_bandpassfilter(template_data(:,:,trial_idx),eeg.srate,input_param.filter,4,'but');
                end
               
                
                if (eeg.window(1)<= input_param.window_len(1) & eeg.window(end) >= input_param.window_len(end))
                    
                    window_size = [input_param.window_len(1)/1000*eeg.srate:input_param.window_len(end)/1000*eeg.srate]-eeg.srate*eeg.window(1)/1000;
                    window_size = floor(window_size);
                    window_size(1) = [];
                    Y_IT(:,:,class_idx,fold_idx) = mean(template_data(ismember(eeg.chan_locs,input_param.ch_list),...
                        window_size,...
                        :),3);
                    
                   W_trca(:,:,class_idx,fold_idx) = trca_weight(template_data(ismember(eeg.chan_locs,input_param.ch_list),...
                        window_size,...
                        :));
                end
                
            end
        end
        
    case 'CITCCA'
        time = input_param.window_len(1)/1000:1/eeg.srate:input_param.window_len(end)/1000;
        time(end) = [];
        
        for har_idx = 1: input_param.harmonics
            for class_idx = 1 : length(eeg.freq)
              Y([1:2]+2*(har_idx-1),:,class_idx) = ...
                  [sin(2*har_idx*eeg.freq(class_idx)*pi*time);...
                  cos(2*har_idx*eeg.freq(class_idx)*pi*time)];
            end
        end
        
        
          
        train_list=nchoosek(1:size(eeg.data,4),input_param.fold-1);
        for i = 1 : size(train_list,1)
           temp_test_list = 1:size(eeg.data,4);
           temp_test_list(train_list(i,:)) = [];
           test_list(i,:) = temp_test_list;
        end
        
        for class_idx = 1 : length(eeg.freq)
            for fold_idx = 1 : size(train_list,1)
                template_data=squeeze(eeg.data(:,:,class_idx,train_list(fold_idx,:)));
                
                template_data = reref(template_data,find(ismember(eeg.chan_locs,input_param.reref_ch)),'keepref','on');
                for trial_idx = 1 : size(template_data,3)
                    template_data(:,:,trial_idx) = ...
                        ft_preproc_bandpassfilter(template_data(:,:,trial_idx),eeg.srate,input_param.filter,4,'but');
                end
                
                if (eeg.window(1)<= input_param.window_len(1) & eeg.window(end) >= input_param.window_len(end))
                    
                    window_size = [input_param.window_len(1)/1000*eeg.srate:input_param.window_len(end)/1000*eeg.srate]-eeg.srate*eeg.window(1)/1000;
                    window_size = floor(window_size);
                    window_size(1) = [];
                    Y_IT(:,:,class_idx,fold_idx) = mean(template_data(ismember(eeg.chan_locs,input_param.ch_list),...
                        window_size,...
                        :),3);
                end
                
            end
        end
        
end




switch (input_param.method)
    
    case 'stdCCA'
        data = reshape(eeg.data,[size(eeg.data,1),size(eeg.data,2),size(eeg.data,3)*size(eeg.data,4)]);
        for class_idx = 1 : length(eeg.freq)
            true_labels_temp(class_idx,:) = class_idx*ones(1,size(eeg.data,[4]));
        end
        true_labels = reshape(true_labels_temp,1,[size(true_labels_temp,1)*size(true_labels_temp,2)]);
        
        %re - ref
        reref_data = reref(data,find(ismember(eeg.chan_locs,input_param.reref_ch)),'keepref','on');
        % filter
        for i = 1 : size(reref_data,3)
            filt_data(:,:,i) = ft_preproc_bandpassfilter(reref_data(:,:,i),eeg.srate,input_param.filter,4,'but');
        end
        % chan_selection & epcohing
        if (eeg.window(1)<= input_param.window_len(1) & eeg.window(end) >= input_param.window_len(end))
            
           window_size = [input_param.window_len(1)/1000*eeg.srate:input_param.window_len(end)/1000*eeg.srate]-eeg.srate*eeg.window(1)/1000;
           window_size = floor(window_size);
           window_size(1) = [];
            X = filt_data(ismember(eeg.chan_locs,input_param.ch_list),...
                 window_size ...
                ,:);
        else
            disp('check window size');
        end
      
        
        
    case 'FBCCA'
        
        data = reshape(eeg.data,[size(eeg.data,1),size(eeg.data,2),size(eeg.data,3)*size(eeg.data,4)]);
        for class_idx = 1 : length(eeg.freq)
            true_labels_temp(class_idx,:) = class_idx*ones(1,size(eeg.data,[4]));
        end
        true_labels = reshape(true_labels_temp,1,[size(true_labels_temp,1)*size(true_labels_temp,2)]);
        
        %re - ref
        reref_data = reref(data,find(ismember(eeg.chan_locs,input_param.reref_ch)),'keepref','on');
        
        % filter
        for i = 1 : size(reref_data,3)
            for har_idx = 1 : input_param.harmonics
                filt_data(:,:,i,har_idx) = ...
                    ft_preproc_bandpassfilter(reref_data(:,:,i),eeg.srate,input_param.filter_bank(har_idx,:),4,'but');
            end
        end
        % chan_selection & epcohing
        if (eeg.window(1)<= input_param.window_len(1) & eeg.window(end) >= input_param.window_len(end))
            
           window_size = [input_param.window_len(1)/1000*eeg.srate:input_param.window_len(end)/1000*eeg.srate]-eeg.srate*eeg.window(1)/1000;
           window_size = floor(window_size);
           window_size(1) = [];
            X = filt_data(ismember(eeg.chan_locs,input_param.ch_list),...
                 window_size ...
                ,:,:);
        else
            disp('check window size');
        end
        
    case 'ITCCA'
        
        
        for class_idx = 1 : length(eeg.freq)
            for fold_idx = 1 : size(test_list,1)
                test_data=squeeze(eeg.data(:,:,class_idx,test_list(fold_idx,:)));
                
                test_data = reref(test_data,find(ismember(eeg.chan_locs,input_param.reref_ch)),'keepref','on');
                for trial_idx = 1 : size(test_data,3)
                    test_data(:,:,trial_idx) = ...
                        ft_preproc_bandpassfilter(test_data(:,:,trial_idx),eeg.srate,input_param.filter,4,'but');
                end
                
                if (eeg.window(1)<= input_param.window_len(1) & eeg.window(end) >= input_param.window_len(end))
                    
                    window_size = [input_param.window_len(1)/1000*eeg.srate:input_param.window_len(end)/1000*eeg.srate]-eeg.srate*eeg.window(1)/1000;
                    window_size = floor(window_size);
                    window_size(1) = [];
                    X_test_temp(:,:,:,class_idx,fold_idx) = test_data(ismember(eeg.chan_locs,input_param.ch_list),...
                        window_size,...
                        :);
                    
                   
                end
                
            end
        end
        
         X_test = permute(X_test_temp,[1,2, 4,5,3]);
        
        for class_idx = 1 : size(X_test,3)
            for fold_idx = 1 : size(X_test,4)
                for test_idx = 1 : size(X_test,5)
                   true_labels(class_idx,fold_idx,test_idx) = class_idx*ones(1,1);
                end
            end
        end

        
        
    case 'TRCA'
               
        for class_idx = 1 : length(eeg.freq)
            for fold_idx = 1 : size(test_list,1)
                test_data=squeeze(eeg.data(:,:,class_idx,test_list(fold_idx,:)));
                
                test_data = reref(test_data,find(ismember(eeg.chan_locs,input_param.reref_ch)),'keepref','on');
                for trial_idx = 1 : size(test_data,3)
                    test_data(:,:,trial_idx) = ...
                        ft_preproc_bandpassfilter(test_data(:,:,trial_idx),eeg.srate,input_param.filter,4,'but');
                end
                
                if (eeg.window(1)<= input_param.window_len(1) & eeg.window(end) >= input_param.window_len(end))
                    
                    window_size = [input_param.window_len(1)/1000*eeg.srate:input_param.window_len(end)/1000*eeg.srate]-eeg.srate*eeg.window(1)/1000;
                    window_size = floor(window_size);
                    window_size(1) = [];
                    X_test_temp(:,:,:,class_idx,fold_idx) = test_data(ismember(eeg.chan_locs,input_param.ch_list),...
                        window_size,...
                        :);
                    
                   
                end
                
            end
        end
        
         X_test = permute(X_test_temp,[1,2, 4,5,3]);
        
        for class_idx = 1 : size(X_test,3)
            for fold_idx = 1 : size(X_test,4)
                for test_idx = 1 : size(X_test,5)
                   true_labels(class_idx,fold_idx,test_idx) = class_idx*ones(1,1);
                end
            end
        end 
        
        
        
    case 'CITCCA'
        
        for class_idx = 1 : length(eeg.freq)
            for fold_idx = 1 : size(test_list,1)
                test_data=squeeze(eeg.data(:,:,class_idx,test_list(fold_idx,:)));
                
                test_data = reref(test_data,find(ismember(eeg.chan_locs,input_param.reref_ch)),'keepref','on');
                for trial_idx = 1 : size(test_data,3)
                    test_data(:,:,trial_idx) = ...
                        ft_preproc_bandpassfilter(test_data(:,:,trial_idx),eeg.srate,input_param.filter,4,'but');
                end
                
                if (eeg.window(1)<= input_param.window_len(1) & eeg.window(end) >= input_param.window_len(end))
                    
                    window_size = [input_param.window_len(1)/1000*eeg.srate:input_param.window_len(end)/1000*eeg.srate]-eeg.srate*eeg.window(1)/1000;
                    window_size = floor(window_size);
                    window_size(1) = [];
                    X_test_temp(:,:,:,class_idx,fold_idx) = test_data(ismember(eeg.chan_locs,input_param.ch_list),...
                        window_size,...
                        :);
                    
                   
                end
                
            end
        end
        
         X_test = permute(X_test_temp,[1,2,4,5,3]);
        
        for class_idx = 1 : size(X_test,3)
            for fold_idx = 1 : size(X_test,4)
                for test_idx = 1 : size(X_test,5)
                   true_labels(class_idx,fold_idx,test_idx) = class_idx*ones(1,1);
                end
            end
        end
end

% classification
switch (input_param.method)
    
    case 'stdCCA'
        for trial_idx = 1 : size(X,3)
            clear r_vec;
            for class_idx = 1: length(eeg.freq)
                [~,~,r] = canoncorr(X(:,:,trial_idx)',Y(:,:,class_idx)');
                r_vec(class_idx) = max(r);
%                 r_vec2(class_idx) = sum(r);
                
            end
            [~,predict_labels(trial_idx)] = max(r_vec);
%             [~,predict_labels2(trial_idx)] = max(r_vec);
        end
        
        
        
        acc(subj_idx) = sum(true_labels == predict_labels)/length(true_labels);
        itr(subj_idx) = Wolpaw_ITR(acc(subj_idx),length(eeg.freq),0.5 +input_param.window_len(end)/1000);
%         acc2 = sum(true_labels == predict_labels2)/length(true_labels);
        
        disp(sprintf('subject:%02d   Acc:%04f %%  ITR:%04f bits/min',subj_idx,100*acc(subj_idx),itr(subj_idx)));
    case 'FBCCA'
        
        for filt_idx = 1 : size(input_param.filter_bank,1)
            weight_n(filt_idx) = filt_idx^(-1*input_param.weight_ab(1)) + input_param.weight_ab(end);
        end
        
        for trial_idx =  1: size(X,3)
            clear r_mat;
            for class_idx = 1: length(eeg.freq)
                for filt_idx = 1 : size(input_param.filter_bank,1)
                    [~,~,r] = canoncorr(X(:,:,trial_idx,filt_idx)',Y(:,:,class_idx)');
                    r_mat(class_idx,filt_idx) = max(r);
                end
            end
            
            [~, predict_labels(trial_idx) ]= max(((r_mat).^2)*weight_n');
        end
        acc(subj_idx) = sum(true_labels == predict_labels)/length(true_labels);
        itr(subj_idx) = Wolpaw_ITR(acc(subj_idx),length(eeg.freq),0.5 +input_param.window_len(end)/1000);
%         acc2 = sum(true_labels == predict_labels2)/length(true_labels);
        
        disp(sprintf('subject:%02d   Acc:%04f %%  ITR:%04f bits/min',subj_idx,100*acc(subj_idx),itr(subj_idx)));
        
    case 'ITCCA'
        for trial_idx = 1: size(X_test,3)
            for fold_idx = 1 :size(X_test,4)
                for test_idx = 1 : size(X_test,5)
                    clear r_vec;
                    for class_idx = 1 : length(eeg.freq)
                        [~,~,r] = canoncorr(X_test(:,:,trial_idx,fold_idx,test_idx)',Y_IT(:,:,class_idx,fold_idx)');

                        r_vec(class_idx) = max(r);
                    end
                   
                    [~,predict_labels(trial_idx,fold_idx,test_idx)] = max(r_vec);
                end
            end
        end
        
%         for i = 1 : 3
%             check_x(:,:,i) = magic(5);
%             check_y(:,:,i) = magic(5);
%         end
%        squeeze( sum(check_x==check_y,1))        
        
%         predict_labels = reshape(predict_labels,)
        temp_acc = squeeze(sum(true_labels == predict_labels,1)/length(true_labels));
        if iscolumn(temp_acc')
            temp_acc = temp_acc';
        end
        acc(subj_idx,:) = mean(temp_acc,2);
        for fold_idx =1: size(acc,2)
            itr(subj_idx,fold_idx) = Wolpaw_ITR(acc(subj_idx,fold_idx),length(eeg.freq),0.5 +input_param.window_len(end)/1000);
        end
        
        disp(sprintf('subject:%02d   Acc:%04f %%  ITR:%04f bits/min',subj_idx,mean(acc(subj_idx,:),2)*100,mean(itr(subj_idx),2)));
        
    case 'TRCA'
        
        for trial_idx = 1: size(X_test,3)
            for fold_idx = 1 :size(X_test,4)
                for test_idx = 1 : size(X_test,5)
                    clear r_vec;
                    for class_idx = 1 : length(eeg.freq)
                        input_X = X_test(:,:,trial_idx,fold_idx,test_idx)'*W_trca(:,:,class_idx,fold_idx);
                        input_Y = Y_IT(:,:,class_idx,fold_idx)'*W_trca(:,:,class_idx,fold_idx);
                        
                        [~,~,r] = canoncorr(input_X,input_Y);

                        r_vec(class_idx) = max(r);
                    end
                   
                    [~,predict_labels(trial_idx,fold_idx,test_idx)] = max(r_vec);
                end
            end
        end
        
%         for i = 1 : 3
%             check_x(:,:,i) = magic(5);
%             check_y(:,:,i) = magic(5);
%         end
%        squeeze( sum(check_x==check_y,1))        
        
%         predict_labels = reshape(predict_labels,)
        temp_acc = squeeze(sum(true_labels == predict_labels,1)/length(true_labels));
        if iscolumn(temp_acc')
            temp_acc = temp_acc';
        end
        acc(subj_idx,:) = mean(temp_acc,2);
        for fold_idx =1: size(acc,2)
            itr(subj_idx,fold_idx) = Wolpaw_ITR(acc(subj_idx,fold_idx),length(eeg.freq),0.5 +input_param.window_len(end)/1000);
        end
        
        disp(sprintf('subject:%02d   Acc:%04f %%  ITR:%04f bits/min',subj_idx,mean(acc(subj_idx,:),2)*100,mean(itr(subj_idx),2)));
        
        
    case 'CITCCA'
        for trial_idx = 1: size(X_test,3)
            for fold_idx = 1 :size(X_test,4)
                for test_idx = 1 : size(X_test,5)
                    clear r_mat;
                    for class_idx = 1 : length(eeg.freq)
                        % X, Y
                        clear Wx Wy ;
                        [Wx,Wy,r1] = canoncorr(X_test(:,:,trial_idx,fold_idx,test_idx)',Y(:,:,class_idx)');
                        
                        r_mat(class_idx,1) = max(r1);
                        
                        % X, Y_template
                        clear Wx Wy;
                        [Wx,Wy,~] = canoncorr(X_test(:,:,trial_idx,fold_idx,test_idx)',Y_IT(:,:,class_idx,fold_idx)');
                        [r2,~] = corr(X_test(:,:,trial_idx,fold_idx,test_idx)'*Wx,Y_IT(:,:,class_idx,fold_idx)' *Wx,'type','Pearson' );
                        r_mat(class_idx,2) = max((r2),[],'all');
                        
                        % X,Y
                        clear Wx Wy;
                        [Wx,Wy,~] = canoncorr(X_test(:,:,trial_idx,fold_idx,test_idx)',Y(:,:,class_idx)');
                        [r3,~] = corr(X_test(:,:,trial_idx,fold_idx,test_idx)'*Wx, Y_IT(:,:,class_idx,fold_idx)'*Wx,'type','Pearson');
                        r_mat(class_idx,3) = max((r3),[],'all');
                        
                        % Y_template, Y
                        clear Wx Wy;
                        [Wx,Wy,~] = canoncorr(Y_IT(:,:,class_idx,fold_idx)',Y(:,:,class_idx)');
                        [r4,~] = corr(X_test(:,:,trial_idx,fold_idx,test_idx)'*Wx,Y_IT(:,:,class_idx,fold_idx)'*Wx,'type','Pearson');
                        r_mat(class_idx,4) = max((r4),[],'all');
                        

                        
                    end
%                     sum(r_mat.^2,2)
                    [~,predict_labels(trial_idx,fold_idx,test_idx)] = max(sum(r_mat.^2,2));
                end
            end
        end
        
%         for i = 1 : 3
%             check_x(:,:,i) = magic(5);
%             check_y(:,:,i) = magic(5);
%         end
%        squeeze( sum(check_x==check_y,1))        
        
%         predict_labels = reshape(predict_labels,)
        temp_acc = squeeze(sum(true_labels == predict_labels,1)/length(true_labels));
        if iscolumn(temp_acc')
            temp_acc = temp_acc';
        end
        acc(subj_idx,:) = mean(temp_acc,2);
        for fold_idx =1: size(acc,2)
            itr(subj_idx,fold_idx) = Wolpaw_ITR(acc(subj_idx,fold_idx),length(eeg.freq),0.5 +input_param.window_len(end)/1000);
        end
        
        disp(sprintf('subject:%02d   Acc:%04f %%  ITR:%04f bits/min',subj_idx,mean(acc(subj_idx,:),2)*100,mean(itr(subj_idx),2)));
        
        
        
end

results.method = input_param.method;
results.window_len = input_param.window_len;
results.chan = input_param.ch_list;
results.acc = acc;
results.itr = itr;

switch input_param.method
    case 'stdCCA'
        results.filter = input_param.filter;
    case 'FBCCA'
        results.filter = input_param.filter_bank;
        results.weight = input_param.weight_ab;
    case 'ITCCA'
        results.filter = input_param.filter;
        results.fold = input_param.fold;
    case 'TRCA'
        results.filter = input_param.filter;
        results.fold = input_param.fold;
    case 'CITCCA'
        results.filter = input_param.filter;
        results.fold = input_param.fold;
end


save(sprintf('%s/Performance_%s.mat',save_path,input_param.method),'results','-v6');

