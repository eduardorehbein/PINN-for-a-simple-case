% Clears workspace
clear all
clc

% Sets parameters
R = 3;
L = 3;

simTime = 7;
simStepSize = 1e-2;

stepTime = 0;
v0 = 0;

% Simulates circuit
data = 0;
for j = -9:2:9
   vf = j;
   sim('circuito_RL');
   if data == 0
       data = [i.Time i.Data v.Data];
   else
       data = [data; i.Time i.Data v.Data];
   end
   
   clear i v
end

% Prints data to csv
headers = {'t','i','v'};
T = array2table(data);
T.Properties.VariableNames(1:3) = headers;
writetable(T, 'multi_pinn.csv');