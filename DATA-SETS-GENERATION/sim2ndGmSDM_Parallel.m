%% Parallel simulation of a 2nd-order GmC SDM (GmC2ndCTSDMParam.mdl)
% J.M. de la Rosa (created April 29 / 2022; modified May 5 / 2022)
% Running multiple simulations in parallel
% Model parameters loaded in the SIMULINK model (see GmCSDMparameters.m)
close all;clc;clear;
tStart1 = cputime;
%% Prepare Simulation Parameters Inputs
SDMmodel = 'GmC2ndCTSDMParam';
load_system(SDMmodel);
Gm2ndCTSDMparameters;
%k=0;npAdc=40;npgb=25;npip=10;
k=0;npAdc=5;npgb=2;npip=2;
nptotal=5*npAdc*npgb*npip;
n=0;np=0;k=0;l=0;m=0;
SDin(1:5, 1:npAdc, 1:npgb, 1:npip) = Simulink.SimulationInput(SDMmodel);

for OSR=[32 64 128 256 512]
n=n+1;
M=OSR;
BW(n)=fs/(2*OSR);
k=0;
for Adc=logspace(1,3,npAdc),
    k=k+1;
    l=0;
    for GBW=logspace(6,9,npgb),
        l=l+1;
        m=0;
        for IIP3dBm=linspace(-10,50,npip),
            m=m+1;
            Ao(k)=Adc;gb(l)=GBW;iip3(m)=IIP3dBm;
%fin=BW(n)/5*0.991234223;
%fin=BW(n)/7*0.991234223;
GBW1=GBW*(1+(rand-0.5)*0.5);
gbw1(l)=GBW1;
Cp1=gm11/(2*pi*GBW1); %% Cp1
gm1(l)=gbw1(l)*1e-12; % Gm used to estimate power GBW/CL (CL=1pF)
GBW2=GBW*(1+(rand-0.5)*0.5);
gbw2(l)=GBW2;
gm2(l)=gbw2(l)*1e-12; % Gm used to estimate power GBW/CL (CL=1pF)
Cp2=gm12/(2*pi*GBW2); %% Cp2
Adc11=Adc*(1+(rand-0.5)*0.5); %% Adc11
Ao1(k)=Adc11;
R11=Adc11/gm11; %% R11
Adc12=Adc*(1+(rand-0.5)*0.5); %% Adc12
R12=Adc12/gm12; %% R12
Ao2(k)=Adc12;
IIP3=1e-3*10^(IIP3dBm/10);
IIP3_in=IIP3*(1+(rand-0.5)*0.5); %% IIP3_in
iip3in(m)=IIP3_in;
IIP3_12=IIP3*(1+(rand-0.5)*0.5); %% IIP3_12
iip3g2(m)=IIP3_12;
IIP3_a=IIP3; %% IIP3_a
iip3ff(m)=IIP3_a;
IIP3_b=IIP3; %% IIP3_b
IIP3_c=IIP3; %% IIP3_c

CompProgress=(np/nptotal)*100;
fprintf('Simulation Parameters Generation: Progress (%%) = %6.2f\n',CompProgress) 

np=np+1;
%SDin(n,k,l,m) = SDin(n,k,l,m).setVariable('fin', fin); 
SDin(n,k,l,m) = SDin(n,k,l,m).setVariable('Cp1', Cp1); 
SDin(n,k,l,m) = SDin(n,k,l,m).setVariable('Cp2', Cp2); 
SDin(n,k,l,m) = SDin(n,k,l,m).setVariable('Adc11', Adc11);
SDin(n,k,l,m) = SDin(n,k,l,m).setVariable('Adc12', Adc12);
SDin(n,k,l,m) = SDin(n,k,l,m).setVariable('R11', R11);
SDin(n,k,l,m) = SDin(n,k,l,m).setVariable('R12', R12);
SDin(n,k,l,m) = SDin(n,k,l,m).setVariable('IIP3_in', IIP3_in);
SDin(n,k,l,m) = SDin(n,k,l,m).setVariable('IIP3_12', IIP3_12);
SDin(n,k,l,m) = SDin(n,k,l,m).setVariable('IIP3_a', IIP3_a);
SDin(n,k,l,m) = SDin(n,k,l,m).setVariable('IIP3_b', IIP3_b);
SDin(n,k,l,m) = SDin(n,k,l,m).setVariable('IIP3_c', IIP3_c);
        end
    end
end
end 
clc;
CPUTimeParam = cputime - tStart1
%% Run parallel simulations
tStart2 = cputime;
fprintf('Running parallel simulations')
SDout=parsim(SDin); 
%fprintf('Computing SNR of simulated SDMs')
% for k=1:npgb
%     sn(k)=fsnr(SDout(k).out, 1, N, fs, fin, BW, 30, 30, 1, 1, 1);
% end
CPUTimeSim = cputime - tStart2
%% Compute performance metrics
clc;
tStart3 = cputime;
fprintf('Computing performance metrics (SNR, SNDR, FOM...)')
for n=1:5,
k=0;
for k=1:npAdc
    l=0;
    for l=1:npgb
        m=0;
        for m=1:npip
OSR=[32 64 128 256 512]
% sn(k)=SNRArray
BW(n)=fs/(2*OSR(n));
%fin=BW(n)/7*0.991234223;
sndr(n,k,l,m) = fsnr(SDout(n,k,l,m).out, 1, N, fs, fin, BW(n), 30, 30, 1, 1, 2);
sn(n,k,l,m) = fsnr(SDout(n,k,l,m).out, 1, N, fs, fin, BW(n), 30, 30, 1, 1, 1);
fprintf('SNR(dB)=%6.1f\n',sn(n,k,l,m))
fprintf('SNDR(dB)=%6.1f\n',sndr(n,k,l,m))
% Estimation of the power consumption based on GBW
% A load capacitor of CL=1pF is assumed, and gm=GBW/CL
Pot(n,k,l,m)=(gm1(l)+gm2(l))*Vr^2;
PotmW(n,k,l,m)=Pot(n,k,l,m)*1e3;
fprintf('Power(mW)=%6.2f\n',PotmW(n,k,l,m))
% Schreier FOMS (based on SNDR)
       FOMS(n,k,l,m)=sndr(n,k,l,m)+10*log10(BW(n)/Pot(n,k,l,m));
       fprintf('FOMS(dB)=%6.2f\n',FOMS(n,k,l,m))
               end
    end
end
end 
clc;
CPUTimeMetrics = cputime - tStart3