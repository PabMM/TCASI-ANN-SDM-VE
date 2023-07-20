from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import manhattan_distances
from pandas import read_csv
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from scipy.stats import gaussian_kde

# load dataset
model_name = '2orSCSDM'
file_name = 'DATA-SETS/data_'+model_name+'.csv'
df = read_csv(file_name)
df_train,df_val = train_test_split(df,test_size=0.2, random_state=1)
print(df_val.shape)
# Assume X is your original dataset and rows_to_compare are the rows to compare in X
X_train = df_train[['SNR','OSR','Power']].values
X_val = df_val[['SNR','OSR','Power']].values
# Normalizing X using MinMaxScaler

scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train)
X_val_norm = scaler.transform(X_val)


def look_at_table(look):

    # Calculating the Manhattan distance between the normalized rows and rows to compare
    distances = manhattan_distances(X_train_norm, look)

    # Index of the row in X with the minimum distance to each row to compare
    closest_row_indices = distances.argmin(axis=0)

    # The rows in X that are closest to the rows to compare

    closest_rows = X_train[closest_row_indices]
    return closest_rows



def random_factor(value, range_val=0.05):
    alpha = random.uniform(-range_val, range_val)
    return value * (1 + alpha)
    
random_factor_vec = np.vectorize(random_factor)

def create_histogram(data, bins=10, edgecolor='black', title='Histogram',file_name = '',file_path='Lookuptable_'):
    # Create histogram
    plt.hist(data, bins=bins, edgecolor=edgecolor)

    # Set labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(title)
   # plt.savefig('LOOK-UP-TABLE/'+file_path+file_name+'_.pdf')
    # Show the histogram
    plt.pause(1)



num_iterations = 10
num_rows = X_val.shape[0]


X_val_var = X_val_norm
E_SNR = np.zeros((num_rows,num_iterations))
E_P   = np.zeros((num_rows,num_iterations))
E_FOM = np.zeros((num_rows,num_iterations))

FOM_asked = X_val[:,0]+10*np.log10(2e4/X_val[:,2])

print('Looking up the table')
for i in tqdm(range(num_iterations)):
    range = 0.10
    if not(i==0):
        X_val_var[:,0] =  random_factor_vec(X_val_norm[:,0],range_val=range)  
        X_val_var[:,2] =  random_factor_vec(X_val_norm[:,2],range_val=range)  
    X_predict = look_at_table(X_val_var)
    FOM_sim = X_predict[:,0]+10*np.log10(2e4/X_predict[:,2])
    E = (X_predict-X_val)/X_val
    E_SNR[:,i] =  E[:,0]
    E_P[:,i] =  E[:,2]
    E_FOM[:,i] = (FOM_sim-FOM_asked)/FOM_asked

E_FOM_final = E_FOM.max(axis=1)
indices = E_SNR.argmax(axis=1)
E_SNR_final = np.zeros_like(E_FOM_final)
E_P_final = np.zeros_like(E_FOM_final)
i = 0
for j in indices:
    E_SNR_final[i] = E_SNR[i,j]
    E_P_final[i] = E_P[i,j]
    i += 1


create_histogram(E_FOM_final,bins=100,title="Deviation between FOM and FOM'",file_name='FOM')
alpha = 0.05
print('P(E > {}) = {:.2f} %'.format(alpha, np.mean(E_FOM_final>alpha) * 100))


#create_histogram(E_SNR_final,bins=100,title='Deviation between SNR and SNR',file_name='SNR')
#create_histogram(E_P_final,bins=100,title='Deviation between Power and Power',file_name='Power')

# Now load ANN's results for comparation
plt.close()
E_FOM_ANN = read_csv('LOOK-UP-TABLE/err_fom_SC_2orSC_.csv').values
create_histogram(E_FOM_ANN,bins=100,title="Deviation between FOM and FOM'",file_name='FOM',file_path='2orSC_SC_10_')
print('P(E > {}) = {:.2f} %'.format(alpha, np.mean(E_FOM_ANN>alpha) * 100))

def compare_histogram(data1,data2, bins=10, edgecolor='black', title='Histogram',file_name = '',file_path='Lookuptable_'):
    # Create histogram
    plt.hist(data2, bins=bins, edgecolor=edgecolor)
    plt.hist(data1, bins=bins,edgecolor=edgecolor,linewidth = 0.3)

    # Set labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend(['Look up Table','ANN'])
   # plt.savefig('LOOK-UP-TABLE/'+file_path+file_name+'_.pdf')
    # Show the histogram
    plt.pause(1)

compare_histogram(E_FOM_final,E_FOM_ANN,bins=100,title="Deviation between FOM and FOM'",file_name='FOM',file_path='LUTvsANN')