% Clears workspace
clear all
clc

% Sets parameters
R = 3;
L = 3;

simTime = 12;
simStepSize = 1e-3;

stepTime = 6;
v0 = 10;
vf = 0;

noisePower = 1e-6;

% Instances data variable
% data = [0 0 0];

% for j = 1:100
%     % Resets current and voltage variables
%     i = 0;
%     v = 0;
% 
%     % Generates random amplitude and angular frequency
%     v_p = 10*rand(1);
%     omega = 100*rand(1);
%     
%     % Sistem simulation
%     sim('circuito_RL');
%     
%     % Populates data variable
%     data = [data; i.Time i.Data v.Data];
% end
% sim('circuito_RL')
sim('circuito_RL_com_ruido')

plot(i)
data = [i.Time i.Data v.Data];

% Prints data to csv
headers = {'t','noisy_i','v'};
% headers = {'t','i','v'};

T = array2table(data);
T.Properties.VariableNames(1:3) = headers;
% writetable(T,'t_i_v_v3.csv');
writetable(T,'noisy_t_i_v_v3.csv');