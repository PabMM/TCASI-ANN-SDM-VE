modulators = {'2orSC'};
model_names = {'SingleClassGB'};
n_mod = 3;
n_model = 2;

for i =1:n_mod
    for j=1:n_model
        Stats_calculator(modulators{i},model_names{j})
    end
end

function []=Stats_calculator(modulator_name,model_name)

data_path = [cd,'\VAL-DS\sim_',modulator_name,'_',model_name,'_10.mat'];

load(data_path)
num_iterations = 1;
fom_sim = fom_sim(:,1:num_iterations);


SNR_sim = SNR_sim(:,1:num_iterations);
power_sim = power_sim(:,1:num_iterations);


[fom_sim,J] = max(fom_sim,[],2);



aux = fom_sim;
auy = aux;
 i = 1;
for j =1:length(J)

    aux(i,1) = SNR_sim(i,J(j));
    auy(i,1) = power_sim(i,J(j));
    i = i+1;
end
SNR_sim = aux; clear aux
power_sim = auy; clear auy


err_fom = (fom_sim-fom_asked)./fom_asked;
err_SNR = (SNR_sim-SNR_asked)./SNR_asked;
err_power = (power_sim-power_asked)./power_asked;

Stats(err_fom,'FOM',[modulator_name,'  ',model_name])
Stats(err_SNR,'SNR',[modulator_name,'  ',model_name])
Stats(err_power,'power',[modulator_name,'  ',model_name])
end

function Stats(err,name,model_name)

moda = mode(err)*100;
m = mean(err)*100;
s = std(err)*100;
P = mean(err>-.0);


fprintf([model_name,': ',name])
fprintf(': \nMean %.2f, std %.2f, P = %.2f\n',m,s,P)
fprintf('Min = %.3f, Max = %.3f\n',min(err),max(err))
alpha = 1-[0.25,0.50,0.75,0.95];
for i = 1:length(alpha)
    fprintf('Q(%.2f) = %.3f\n',alpha(i),quantile(err,alpha(i)))
end


end