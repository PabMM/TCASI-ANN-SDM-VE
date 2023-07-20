%% Import data

file_name = 'sim_2orSC_SingleClass_10';
model_name = '2orSC_SC';
num_iterations = 10;
load(['VAL-DS/',file_name,'.mat'])

SNR_sim = SNR_sim(:,1:num_iterations);
SNR_sim = max(SNR_sim,[],2);
err = (SNR_sim-SNR_asked)./SNR_asked;

ft=14;
tt=16;

%Graficas
close all
figure (1)
hold on

histogram(err,100)
m = mean(err)*100;
s = std(err)*100;
P = mean(err>-.0);
alpha = 0.05;
Q = quantile(err,alpha);

fprintf(model_name)
fprintf(': Media %.2f, Desviación %.2f, P = %.2f, Q(%.2f) = %.2f\n',m,s,P,alpha,Q)
   

legend('E')
hold off

xlabel('E','FontSize',ft)
ylabel('Number of occurrences','FontSize',ft)
title('Deviation between SNR y SNR^{\prime}','FontSize',tt)

graph_name = ['Images/E_distribution_predictedlabel_',model_name,num2str(num_iterations),'.pdf'];
%exportgraphics(figure (1) ,graph_name,'ContentType','vector')
