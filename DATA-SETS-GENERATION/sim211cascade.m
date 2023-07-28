% Generation of a dataset for a 2nd-ord SC SDM
% P. Diaz January 18,2023 based on J.M. de la Rosa March 14, 2022

clear;clc;close all;
tStart = cputime;

Bw=4e6;
fin=Bw/3;

%% Prepare Simulation Parameters Inputs
SDMmodel = 'umts211_PQ';
load_system(SDMmodel);

% Number of points
% npAdc=10;npgm=10;npio=10;npVn=10;npcs1a=5;
np = 20;
nsim = 40;
% SDin(1:5, 1:npAdc, 1:npgm, 1:npio, 1:npVn,1:npcs1a) = Simulink.SimulationInput(SDMmodel);

% Parameters' values
OSR=[ 128 256 512];
Adc=logspace(1,3,np);
gm=logspace(-5,-3,np);
io=logspace(-4,-2,np);
Vn=logspace(-11,-7,np);
cs1a=logspace(-15,-10,np);
SDin(1:3, 1:nsim, 1:nsim) = Simulink.SimulationInput(SDMmodel);
for k = 1:nsim   
    for n = 1:3
        for l = 1:nsim
            for m = 1:nsim
                M = OSR(n);
                fs = 2*M*Bw;
                ts = 1/fs; 
                
                SDin(n,l,m) = SDin(n,l,m).setVariable('ts', ts);
                SDin(n,l,m) = SDin(n,l,m).setVariable('fs', fs);
                
                
                SDin(n,l,m) = SDin(n,l,m).setVariable('OSR', OSR(n));

                ale = randi([1, np], 1, 12);

                SDin(n,l,m) = SDin(n,l,m).setVariable('Adc1', Adc(ale(1)));
                SDin(n,l,m) = SDin(n,l,m).setVariable('gm1', gm(ale(2)));
                SDin(n,l,m) = SDin(n,l,m).setVariable('io1', io(ale(3)));

                SDin(n,l,m) = SDin(n,l,m).setVariable('Adc2', Adc(ale(4)));
                SDin(n,l,m) = SDin(n,l,m).setVariable('gm2', gm(ale(5)));
                SDin(n,l,m) = SDin(n,l,m).setVariable('io2', io(ale(6)));

                SDin(n,l,m) = SDin(n,l,m).setVariable('Adc3', Adc(ale(4)));
                SDin(n,l,m) = SDin(n,l,m).setVariable('gm3', gm(ale(5)));
                SDin(n,l,m) = SDin(n,l,m).setVariable('io3', io(ale(6)));

                SDin(n,l,m) = SDin(n,l,m).setVariable('Adc4', Adc(ale(4)));
                SDin(n,l,m) = SDin(n,l,m).setVariable('gm4', gm(ale(5)));
                SDin(n,l,m) = SDin(n,l,m).setVariable('io4', io(ale(6)));
            end
        end
    end

    disp(cputime - tStart)
    % 
    % Run parallel simulations
    tStart2 = cputime;
    fprintf('Running parallel simulations')
    SDout=parsim(SDin,'ShowProgress','on','TransferBaseWorkspaceVariables','off',...
        'AttachedFiles','211_Variables.mat',...
        'SetupFcn',@()evalin('base','load 211_Variables.mat')); 
    disp(cputime - tStart2)
    fprintf('Saving Data ...')
    c1 = reshape(arrayfun(@(obj) obj.Variables(3).Value, SDin), [], 1);
    c2 = reshape(arrayfun(@(obj) obj.Variables(4).Value, SDin), [], 1);
    c3 = reshape(arrayfun(@(obj) obj.Variables(5).Value, SDin), [], 1);
    c4 = reshape(arrayfun(@(obj) obj.Variables(6).Value, SDin), [], 1);
    c5 = reshape(arrayfun(@(obj) obj.Variables(7).Value, SDin), [], 1);
    c6 = reshape(arrayfun(@(obj) obj.Variables(8).Value, SDin), [], 1);
    c7 = reshape(arrayfun(@(obj) obj.Variables(9).Value, SDin), [], 1);
    c8 = reshape(arrayfun(@(obj) obj.Variables(10).Value, SDin), [], 1);
    c9 = reshape(arrayfun(@(obj) obj.Variables(11).Value, SDin), [], 1);
    c10 = reshape(arrayfun(@(obj) obj.Variables(12).Value, SDin), [], 1);
    c11 = reshape(arrayfun(@(obj) obj.Variables(13).Value, SDin), [], 1);
    c12 = reshape(arrayfun(@(obj) obj.Variables(14).Value, SDin), [], 1);
    c13 = reshape(arrayfun(@(obj) obj.Variables(15).Value, SDin), [], 1);
    c14 = reshape(arrayfun(@(obj) obj.SNRArray, SDout),[],1);
    data = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14];
    dlmwrite('211_DataSet_V1.csv',data,'-append')
    clear c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 c12 c13 c14 data
    fprintf('Data Saved. %.2f %% Completed\n' ,k/np)



end

disp(cputime - tStart)






