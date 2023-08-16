% Generation of a dataset for a 2nd-ord SC SDM
% P. Diaz 

clear;clc;close all;
tStart = cputime;
Bw=4e6;
fin=Bw/3;

%% Prepare Simulation Parameters Inputs
SDMmodel = 'GmC2ndCTSDMParam';
load_system(SDMmodel);

% Parameters' values
n_sim = 5e4;
OSR=[32 64 128 256 512];
OSR = OSR(randi([1, 5], 1, n_sim));
Adc = 10.^(1+2*rand(1,n_sim));
gm = 10.^(-5+2*rand(1,n_sim));
io = 10.^(-4+2*rand(1,n_sim));


load('2ndGmSDM_Variables.mat')
Adc11 = 10.^(1+2*rand(1,n_sim));
R11=Adc11/gm11;

Adc12 = 10.^(1+2*rand(1,n_sim));
R12=Adc12/gm12;

GBW1 = 10.^(6+3*rand(1,n_sim)); 
Cp1=gm11./(2*pi*GBW1);

GBW2 = 10.^(6+3*rand(1,n_sim));
Cp2=gm12./(2*pi*GBW2);

IIP3_in = -10+60*rand(1,n_sim);
IIP3_12 = -10+60*rand(1,n_sim);
IIP3_FF = -10+60*rand(1,n_sim);

SDin(1:n_sim) = Simulink.SimulationInput(SDMmodel);
for n = 1:n_sim   

M = OSR(n);
fs = 1e9;
BW=fs./(2*M);
% prepare simulation input

% Simulation input
SDin(n) = SDin(n).setVariable('M', M); 
SDin(n) = SDin(n).setVariable('BW', BW);
SDin(n) = SDin(n).setVariable('OSR', M); 
SDin(n) = SDin(n).setVariable('Cp1', Cp1(n)); 
SDin(n) = SDin(n).setVariable('Cp2', Cp2(n)); 
SDin(n) = SDin(n).setVariable('Adc11', Adc11(n));
SDin(n) = SDin(n).setVariable('Adc12', Adc12(n));
SDin(n) = SDin(n).setVariable('R11', R11(n));
SDin(n) = SDin(n).setVariable('R12', R12(n));
SDin(n) = SDin(n).setVariable('IIP3_in', IIP3_in(n));
SDin(n) = SDin(n).setVariable('IIP3_12', IIP3_12(n));
SDin(n) = SDin(n).setVariable('IIP3_a', IIP3_FF(n));
SDin(n) = SDin(n).setVariable('IIP3_b', IIP3_FF(n));
SDin(n) = SDin(n).setVariable('IIP3_c', IIP3_FF(n));
fprintf(['Simulation input creation ',num2str(n/n_sim*100),'\n'])
end

disp(cputime - tStart)
% 
% Run parallel simulations
tStart2 = cputime;
fprintf('Running parallel simulations')
SDout=parsim(SDin,'ShowProgress','on','TransferBaseWorkspaceVariables','off',...
    'AttachedFiles','2ndGmSDM_Variables.mat',...
    'SetupFcn',@()evalin('base','load 2ndGmSDM_Variables.mat')); 
disp(cputime - tStart2)
fprintf('Saving Data ...')


snr = fsnr(out,1,N,fs,fin,fs./(2*OSR),30,30,1,1,1);
data = [snr,OSR,Adc11,Adc12,GBW1,GBW2,IIP3_int1,IIP3_int2,IIP3_FF];

data = array2table(data,'VariableNames',{'SNR', 'OSR', 'Adc11', 'Adc12', 'GBW1','GBW2', 'IIP3_int1', 'IIP3_int2','IIP3_FF'});
writetable(data,'2orGMSDM_DataSet_random.csv','WriteMode','append')


disp(cputime - tStart)






