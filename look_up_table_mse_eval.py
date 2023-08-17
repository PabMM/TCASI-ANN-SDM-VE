from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from scipy.stats import gaussian_kde

# load dataset
model_name = '211CascadeSDM'
file_name = 'DATA-SETS/data_'+model_name+'.csv'
df = read_csv(file_name)

# split into input (x) and output (y) variables
specs_columns = df[['SNR','OSR','Power']].values
design_vars = df.drop(['SNR', 'OSR','Power'], axis=1).values

y_scaler = MinMaxScaler((0,1))
y_scaled = y_scaler.fit_transform(design_vars)

X_train, X_val, Y_train, Y_val = train_test_split(specs_columns, y_scaled, test_size=0.2, random_state=1)



def look_at_table_manhattan(look):

    # Calculating the Manhattan distance between the normalized rows and rows to compare
    distances = manhattan_distances(X_train, look)

    # Index of the row in X with the minimum distance to each row to compare
    closest_row_indices = distances.argmin(axis=0)

    # The rows in X that are closest to the rows to compare

    closest_rows_X_train = X_train[closest_row_indices]
    closest_rows_Y_train = Y_train[closest_row_indices]
    
    return closest_rows_X_train, closest_rows_Y_train

def look_at_table_euclidean(look):

    # Calculating the Manhattan distance between the normalized rows and rows to compare
    distances = euclidean_distances(X_train, look)

    # Index of the row in X with the minimum distance to each row to compare
    closest_row_indices = distances.argmin(axis=0)

    # The rows in X that are closest to the rows to compare

    closest_rows_X_train = X_train[closest_row_indices]
    closest_rows_Y_train = Y_train[closest_row_indices]
    
    return closest_rows_X_train, closest_rows_Y_train



X_predict_manhattan, Y_predict_manhattan = look_at_table_manhattan(X_val)
X_predict_euclidean, Y_predict_euclidean = look_at_table_euclidean(X_val)


# Calcula el MSE entre Y_predict e Y_val
mse_manhattan = mean_squared_error(Y_val, Y_predict_manhattan)
mse_euclidean = mean_squared_error(Y_val, Y_predict_euclidean)

print(f'Modulator:{model_name} MSE_MANHATTAN:{mse_manhattan:.3f} MSE_EUCLID:{mse_euclidean:.3f}')