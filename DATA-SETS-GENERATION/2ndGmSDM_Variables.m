%% Simulation parameters for a 2nd-ord Gm-C SDM

%% SDM Loop Filter coefficients
d=1;
g11=1/4;
g12=1/4;
gq1=4;
gq2=4;
f11=(5/(2*g11))/gq1;
f12=(1/(g11*g12))/gq1;
fi=1/gq1;
kr=2/gq1;
g21=1/4;
g22=1/4;
f21=(5/(2*g11))/gq1;
f22=(1/(g11*g12))/gq1;
fi2=1/gq1;
kr2=2/gq1;

%% Quantizer
vhigh=Vr;
vlow=-Vr;
n_levels=3;

%% GmC Integrators
gm=3e-4;

C1=(gm/fs)/g11;
Ci1=(gm/fs)/g11;
C2=(gm/fs)/g12;

gm11=gm;
Av11dB=1000;
%Adc11=10.^(Av11dB./20);
%R11=Adc11/gm11;

gm12=gm;
Av12dB=1000;
%Adc12=10.^(Av12dB./20);
%R12=Adc12/gm12;

%IIP3_in=1000;
%IIP3_12=1000;
%IIP3_a=1000;
%IIP3_b=1000;
%IIP3_c=1000;

GBW1=10e15;
%Cp1=gm11/(2*pi*GBW1);
GBW2=10e15;
%Cp2=gm12/(2*pi*GBW2);

vos_gm=20;
vis_gm=20;

T=300;

eCT11=0;eCT12=0;eCT21=0;eCT22=0;

