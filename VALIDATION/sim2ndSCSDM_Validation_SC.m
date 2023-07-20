% Validation ANN multiclas for a 2nd-ord SC SDM
% P. Diaz April 19,2023 

clear;clc;close all;
tStart = cputime;

data_path = [cd,'\VAL-DS\Multiple-Iterations-SC\SCANN_2orSCSDM_val_']; % Single Class
% data_path = [cd,'\VAL-DS\Multiple-Iterations-C\Classifier_2orSCSDM_val_']; % Classifier
% Load model
SDMmodel = 'SecondOrderSingleBitSC';
load_system(SDMmodel);
variables_filePath = '2ndSCSDM_Variables.mat';
table = importfile_SC([data_path,num2str(1),'.csv']);

[rows,~]=size(table);
SDin(1:rows) = Simulink.SimulationInput(SDMmodel);

Bw=2e4;
fin=Bw/5;

SNR_asked = table.SNR;
power_asked = table.Power;
fom_asked = SNR_asked+10*log10(Bw./power_asked);

num_iterations = 10;
SNR_sim = zeros(rows,num_iterations);
power_sim = SNR_sim;



for i = 1:num_iterations
    % Read data
    table = importfile_SC([data_path,num2str(i),'.csv']);
    [rows,~]=size(table);
    Adc = table.Adc;
    gm = table.gm;
    io = table.Io;  
    Vn = table.Vn;
    OSR = table.OSR;
    % prepare simulation input
    for n = 1:rows  
          
        M=OSR(n);
        fs=2*M*Bw;
        ts=1/fs; 

        SDin(n) = SDin(n).setVariable('M', OSR(n));
        SDin(n) = SDin(n).setVariable('Adc', Adc(n));
        SDin(n) = SDin(n).setVariable('gm', gm(n));
        SDin(n) = SDin(n).setVariable('io', io(n));
        SDin(n) = SDin(n).setVariable('ts', ts);
        SDin(n) = SDin(n).setVariable('fs', fs);
        
        SDin(n) = SDin(n).setVariable('Vn', Vn(n));
        
                    
    end   

    % 
    % Run parallel simulations
    tStart2 = cputime;
    fprintf('Running parallel simulations')
    SDout=parsim(SDin,'ShowProgress','on','TransferBaseWorkspaceVariables','off',...
        'AttachedFiles',variables_filePath,...
        'SetupFcn',@()evalin('base','load 2ndSCSDM_Variables.mat')); 
    disp(cputime - tStart2)
    fprintf('Saving Data ...')
    
    SNR_sim(:,i) = reshape(arrayfun(@(obj) obj.SNRArray, SDout),[],1);
    power_sim(:,i) = 3.2*io;
    
end
fom_sim = SNR_sim+10*log10(Bw./power_sim);
%%
save(['VAL-DS/sim_2orSC_SingleClass_',num2str(num_iterations),'.mat'],"SNR_asked","SNR_sim","power_sim","power_asked","fom_sim","fom_asked")

function SCANN3orCascadeSDMval29 = importfile_SC(filename, dataLines)


    % If dataLines is not specified, define defaults
    if nargin < 2
        dataLines = [2, Inf];
    end
    
    % Set up the Import Options and import the data
    opts = delimitedTextImportOptions("NumVariables", 8);
    
    % Specify range and delimiter
    opts.DataLines = dataLines;
    opts.Delimiter = ",";
    
    % Specify column names and types
    opts.VariableNames = ["SNR", "OSR", "Power", "Adc", "gm", "Io", "Vn"];
    opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double"];
    
    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";
    
    % Import the data
    SCANN3orCascadeSDMval29 = readtable(filename, opts);
    
    end





