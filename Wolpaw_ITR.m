function y = Wolpaw_ITR(acc, nb_class,nb_command)
    if acc == 1         % When P =1.00, log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1)) = log2(N)
        N = nb_class;   % number of class 
        C= 60/nb_command;  % commend in minute, ex)stimulus period is 4 second -> C = 60/4  (sec/sec);
        B = log2(N);     
        ITR = B*C; 
        y= ITR;
    else
        N = nb_class; % number of class 
        P = acc;      % accuracy, ex) When accuracy is 80%, value of acc is 0.80
        C= 60/nb_command;  % C commend in minute,ex)stimulus period is 4 second -> C = 60/4  (sec/sec);
        B = log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1));
        ITR = B*C;
        y= ITR;
    end
end 
