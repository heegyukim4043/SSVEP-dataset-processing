clc;  clear all;
ft_defaults;
eeglab; close;

%%
clc; clear; close all;
% cd('E:\TGI');
addpath('./functions');

markers_TGI = [1:3] + 20000;
seq_played = [];
nsub = 43;
nrun = 10;
foi = [0.5 55];

for i=1 % :nsub
%     cd('E:\TGI');
    if ~(ismember(i, [22 30]))
        for num=1 :nrun
            fname = sprintf('./Subject/subject %02d/PainRating-run%02d.gdf', i, num);
            eeg = pop_biosig(fname);
            fprintf('\n \n \n sub: %d, run: %d \n \n \n', i, num);
            
            eeg = pop_resample(eeg,512);

%             eeg.data=resample(eeg.data',1,2);
%             eeg.data = eeg.data';
%             eeg.srate = eeg.srate*(1/2);

            if (isstr(eeg.event(1).type))
%                 eeg = mod_event_markers(eeg);
                temp_event_type = {eeg.event.type};
                temp_event_type = erase(temp_event_type,'condition ');
                eeg.event_type = str2double(temp_event_type);
                 eeg.event_latency = [eeg.event.latency];
            else
                eeg.event_type = [eeg.event.type];
                eeg.event_latency = [eeg.event.latency];
            end
            
            eeg.data = reref(eeg.data, [33 34], 'keepref', 'on');
            eeg.data = ft_preproc_bandpassfilter(eeg.data, eeg.srate, foi, 4, 'but');
            eeg.data = ft_preproc_bandstopfilter(eeg.data, eeg.srate, [58 62], 4, 'but');
            eeg.event_latency = eeg.event_latency *1/2; % Downsampling to 512 Hz
            eeg.event_latency(mod(eeg.event_latency, 1) == 0.5)...
                = eeg.event_latency(mod(eeg.event_latency, 1) == 0.5) - 0.5;

            TGI_7s = ismember(eeg.event_type, markers_TGI);
            Val_TGI_7s = eeg.event_type(TGI_7s) - 20000; % Temperature 1~9 steps
            ID_TGI_7s = eeg.event_latency(TGI_7s);

            % Data Concatenating
            data_EEG{num} = [];
            seq_EEG{num} = [];

            for ntrial =1:length(ID_TGI_7s)
                seq_val = []; seq_test_val = [];
                for epoching=1:7
                    begin_epoch = ID_TGI_7s(ntrial)+3*eeg.srate+((epoching-1)*eeg.srate);
                    end_epoch = begin_epoch + eeg.srate-1;

                    data_EEG{num} = cat(3, data_EEG{num}, eeg.data(1:32, begin_epoch:end_epoch));
                    seq_EEG{num} = cat(2, seq_EEG{num}, Val_TGI_7s(ntrial));
                end

                disp(Val_TGI_7s);
            end
           
            % Label ramdomization
%             seq_EEG{num}(seq_EEG{num} == 1) =1;
%             seq_EEG{num}(seq_EEG{num} == 9) =2; % Class label: 0,1,2,3,...

            setnum = size(markers_TGI,2)*7;
            rorder = randperm(setnum);
            learn_data{num} = [];
            learn_seq{num} = [];
            for random_or = 1:setnum
                learn_data{num} = cat(3, learn_data{num}, data_EEG{num}(:,:,rorder(random_or)));
                learn_seq{num} = cat(2, learn_seq{num}, seq_EEG{num}(:, rorder(random_or)));
            end
        end
        % Fold randomization
        x=randperm(10);
        seq_played = cat(2, seq_played, x');
    
        for fold=1:5
            train_EEG{fold} = [];
            test_EEG{fold} = [];
            train_seq{fold} = [];
            test_seq{fold} = [];
            testgroup = [2*mod(fold, 5),2*mod(fold, 5)+1];
            for j = 1:10
                % 8:2 = train:test
                % total 10 trials into 5 fold (random)
                if ismember(j-1, testgroup) == 1
                    test_EEG{fold}=cat(3, test_EEG{fold}, learn_data{x(j)});
                    test_seq{fold}=cat(2, test_seq{fold}, learn_seq{x(j)});
                else
                    train_EEG{fold}=cat(3, train_EEG{fold}, learn_data{x(j)});
                    train_seq{fold}=cat(2, train_seq{fold}, learn_seq{x(j)});
                end
            end
        end


%         cd('E:\TGI\EEGNET code\data_deep\labelsrand_5fold_TGI\');
        
        subname = sprintf('proc_dat/subject%02d_traindata.mat', i);
        testname = sprintf('proc_dat/subject%02d_testdata.mat', i);
        seqname = sprintf('proc_dat/subject%02d_trainseq.mat', i);
        setrname = sprintf('proc_dat/subject%02d_testseq.mat', i);
        save(subname, 'train_EEG');
        save(seqname, 'train_seq');
        save(testname, 'test_EEG');
        save(setrname, 'test_seq');
        save('random_seq.mat', 'seq_played');

    end
end
