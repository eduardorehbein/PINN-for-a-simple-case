% Clears workspace
clear all
clc

% Sets parameters
R = 3;
L = 3;

simTime = 30;
simStepSize = 1e-2;

noisePower = 1e-6;

% Runs_simulation
sim('noisy_RL_circuit')
plot(noisy_i)

% Prints data to csv
data = [i.Time v.Data i.Data noisy_i.Data];
headers = {'t', 'v', 'i', 'noisy_i'};
T = array2table(data);
T.Properties.VariableNames(1:4) = headers;
writetable(T,'noisy_t_i_v_v4.csv');
