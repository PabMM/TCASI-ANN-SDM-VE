% Validation ANN multiclas for a 3nd-ord SC SDM
% P. Diaz May 23,2023 

clear;clc;close all;
tStart = cputime;

Bw = 1e4;
data_path = [cd,'\VAL-DS\Multiple-Iterations-C\classifierGB_3orCascadeSDM_val_'];
% Load model
SDMmodel = 'ThirdOrderCascadeSingleBitSC';
load_system(SDMmodel);
variables_filePath = '3rdSCSDM_Variables.mat';
% number of rows
table = importfile_SC([data_path,num2str(1),'.csv']);

[rows,~]=size(table);

SDin(1:rows) = Simulink.SimulationInput(SDMmodel);


num_iterations = 10;
SNR_asked = table.SNR;
power_asked = table.Power;
fom_asked = SNR_asked+10*log10(Bw./power_asked);

SNR_sim = zeros(rows,num_iterations);
power_sim = SNR_sim;


for i = 1:num_iterations
    % Read data
    table = importfile_SC([data_path,num2str(i),'.csv']);
    [rows,~]=size(table);
    ao1 = table.Adc1;
    gm1 = table.gm1;
    io1 = table.Io1;  
    ao2 = table.Adc2;
    gm2 = table.gm2;
    io2 = table.Io2;  
    
    OSR = table.OSR;
    % prepare simulation input
    for n = 1:rows  
          
        M=OSR(n);
        fs=2*M*Bw;
        Ts=1/fs; 
        SDin(n) = SDin(n).setVariable('M', OSR(n));
        SDin(n) = SDin(n).setVariable('ao1', ao1(n));
        SDin(n) = SDin(n).setVariable('gm1', gm1(n));
        SDin(n) = SDin(n).setVariable('io1', io1(n));
        SDin(n) = SDin(n).setVariable('ao2', ao2(n));
        SDin(n) = SDin(n).setVariable('gm2', gm2(n));
        SDin(n) = SDin(n).setVariable('io2', io2(n));
        SDin(n) = SDin(n).setVariable('Ts', Ts);
        SDin(n) = SDin(n).setVariable('fs', fs);
        
                    
    end   
   
     
    % Run parallel simulations
    tStart2 = cputime;
    fprintf('Running parallel simulations')
    SDout=parsim(SDin,'ShowProgress','on','TransferBaseWorkspaceVariables','off',...
        'AttachedFiles',variables_filePath,...
        'SetupFcn',@()evalin('base','load 3rdSCSDM_Variables.mat')); 
    disp(cputime - tStart2)
    fprintf('Saving Data ...')
    
    SNR_sim(:,i) = reshape(arrayfun(@(obj) obj.SNRArray, SDout),[],1);
    power_sim(:,i) = 1.52*(io1+io2);
end
%%
fom_sim = SNR_sim+10*log10(Bw./power_sim);
%%
save(['VAL-DS/sim_3orSC_SingleClassGB_',num2str(num_iterations),'.mat'],"SNR_asked","SNR_sim","power_sim","power_asked","fom_sim","fom_asked")


function SCANN3orCascadeSDMval29 = importfile_SC(filename, dataLines)


% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [2, Inf];
end

% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 9);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["SNR", "OSR", "Power", "Adc1", "gm1", "Io1", "Adc2", "gm2", "Io2"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Import the data
SCANN3orCascadeSDMval29 = readtable(filename, opts);

end

