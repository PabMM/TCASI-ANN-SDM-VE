% Generation of a dataset for a 2nd-ord SC SDM
% P. Diaz January 18,2023 based on J.M. de la Rosa March 14, 2022

clear;clc;close all;
tStart = cputime;
Bw = 1e4;


%% Prepare Simulation Parameters Inputs
SDMmodel = 'ThirdOrderCascadeSingleBitSC';
load_system(SDMmodel);

% Parameters' values
n_sim = 5e4;
OSR=[32 64 128 256 512];
OSR = OSR(randi([1, 5], 1, n_sim));

Adc1 = 10.^(1+2*rand(1,n_sim));
gm1 = 10.^(-5+2*rand(1,n_sim));
io1 = 10.^(-4+2*rand(1,n_sim));

Adc2 = 10.^(1+2*rand(1,n_sim));
gm2 = 10.^(-5+2*rand(1,n_sim));
io2 = 10.^(-4+2*rand(1,n_sim));

SDin(1:n_sim) = Simulink.SimulationInput(SDMmodel);
for n = 1:n_sim   
    
    M = OSR(n);
    fs = 2*M*Bw;
    ts = 1/fs; 

    SDin(n) = SDin(n).setVariable('ts', ts);
    SDin(n) = SDin(n).setVariable('fs', fs);

    SDin(n) = SDin(n).setVariable('M', OSR(n));

    SDin(n) = SDin(n).setVariable('ao1', Adc1(n));
    SDin(n) = SDin(n).setVariable('gm1', gm1(n));
    SDin(n) = SDin(n).setVariable('io1', io1(n));

    SDin(n) = SDin(n).setVariable('ao2', Adc2(n));
    SDin(n) = SDin(n).setVariable('gm2', gm2(n));
    SDin(n) = SDin(n).setVariable('io2', io2(n));


    fprintf(['Simulation input creation ',num2str(n/n_sim*100),'\n'])
end

disp(cputime - tStart)
% 
% Run parallel simulations
tStart2 = cputime;
fprintf('Running parallel simulations')
SDout=parsim(SDin,'ShowProgress','on','TransferBaseWorkspaceVariables','off',...
    'AttachedFiles','3rdSCSDM_Variables.mat',...
    'SetupFcn',@()evalin('base','load 3rdSCSDM_Variables.mat')); 
disp(cputime - tStart2)
fprintf('Saving Data ...')
osr = reshape(arrayfun(@(obj) obj.Variables(3).Value, SDin), [], 1);

adc1 = reshape(arrayfun(@(obj) obj.Variables(4).Value, SDin), [], 1);
gm1 = reshape(arrayfun(@(obj) obj.Variables(5).Value, SDin), [], 1);
io1 = reshape(arrayfun(@(obj) obj.Variables(6).Value, SDin), [], 1);

adc2 = reshape(arrayfun(@(obj) obj.Variables(7).Value, SDin), [], 1);
gm2 = reshape(arrayfun(@(obj) obj.Variables(8).Value, SDin), [], 1);
io2 = reshape(arrayfun(@(obj) obj.Variables(9).Value, SDin), [], 1);


snr = reshape(arrayfun(@(obj) obj.SNRArray, SDout),[],1);

data = [snr,osr,adc1,gm1,io1,adc2,gm2,io2];

data = array2table(data,'VariableNames',{'SNR', 'OSR','Power', 'Adc', 'gm1', 'Io1','Adc2', 'gm2', 'Io2','Adc3', 'gm3', 'Io3','Adc4', 'gm4', 'Io4'});
writetable(data,'3or21CascadeSDM_DataSet_random.csv','WriteMode','append')


disp(cputime - tStart)






