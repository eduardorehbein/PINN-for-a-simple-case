% Clears workspace
clear all
clc

% Sets parameters
R = 3;
L = 3;

simTime = 30;
simStepSize = 1e-1;

stepTime = 6;
v0 = 10;
vf = 0;

noisePower = 1e-6;

% Runs_simulation
sim('circuito_RL_com_ruido')
plot(i)

% Prints data to csv
data = [i.Time v.Data i.Data noisy_i.Data];
headers = {'t', 'v', 'i', 'noisy_i'};
T = array2table(data);
T.Properties.VariableNames(1:4) = headers;
writetable(T,'noisy_t_i_v_v4.csv');
