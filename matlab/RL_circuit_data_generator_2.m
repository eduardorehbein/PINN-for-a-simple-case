% Clears workspace
clear all
clc

% Sets parameters
R = 3;
L = 3;

simTime = 7;
simStepSize = 1e-2;

noisePower = 1e-6;

% Runs_simulation
data = [0 0 0 0];
for j = 1:20
    v0 = j;
    sim('noisy_step_v_RL_circuit')
    data = [data; i.Time v.Data i.Data noisy_i.Data];
end
plot(data(:,4))

% Prints data to csv
headers = {'t', 'v', 'i', 'noisy_i'};
T = array2table(data);
T.Properties.VariableNames(1:4) = headers;
writetable(T,'noisy_t_i_v_v7.csv');
