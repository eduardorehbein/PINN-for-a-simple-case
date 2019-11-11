% Clears workspace
clear all
clc

% Sets parameters
R = 1e3;
L = 1e-3;
simTime = 50;

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

v_p = 10;
omega = 100;

sim('circuito_RL');

data = [i.Time i.Data v.Data];

% Prints data to csv
headers = {'t','i','v'};
% csvwrite('t_i_v.csv', data, headers);

T = array2table(data);
T.Properties.VariableNames(1:3) = headers;
writetable(T,'t_i_v_v2.csv');