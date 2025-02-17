function W = trca_weight(data) 

%
% Input:
%   eeg         : Input eeg data 
%                 (# of channels, Data length [sample], # of trials)
%
% Output:
%   W           : Weight coefficients for electrodes which can be used as 
%                 a spatial filter.
%   


[num_chans, num_smpls, num_trials]  = size(data);
S = zeros(num_chans);
for trial_i = 1:1:num_trials
    for trial_j = trial_i+1:1:num_trials
        x1 = squeeze(data(:,:,trial_i));
        x1 = bsxfun(@minus, x1, mean(x1,2));
        x2 = squeeze(data(:,:,trial_j));
        x2 = bsxfun(@minus, x2, mean(x2,2));
        S = S + x1*x2';
    end % trial_j
end % trial_i
UX = reshape(data, num_chans, num_smpls*num_trials);
UX = bsxfun(@minus, UX, mean(UX,2));
Q = UX*UX';
[W,~] = eig(S, Q);