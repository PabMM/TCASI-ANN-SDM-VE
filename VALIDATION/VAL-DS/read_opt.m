modulators = {'2orSC','2orGM','3orSC','211Cascade'};
n = length(modulators);
T = zeros(n,3);
for i = 1:n
    file_path = ['sim_',modulators{i},'_OPTGB_10.mat'];
    load(file_path);
    [T(i,1),I] = max(fom_sim);
    T(i,2) = SNR_sim(I);
    T(i,3) = power_sim(I);

end


T = array2table(T,'VariableNames',{'FOM','SNR','Power'},'RowName',modulators);
disp(T)