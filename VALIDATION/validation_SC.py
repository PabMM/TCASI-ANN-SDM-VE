import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
import pandas as pd
import random

import joblib
from tqdm import tqdm

# load dataset
model_name = '3orCascadeSDM'
file_name = 'DATA-SETS/data_'+model_name+'.csv'

df = read_csv(file_name)
df_train,df_val = train_test_split(df,test_size=0.2,random_state=1)


#load model
model = keras.models.load_model('SINGLECLASS-ANN/models/'+model_name)
model.summary()

# load scaler 
y_scaler = joblib.load('SINGLECLASS-ANN/scalers/model_'+model_name+'_scaler.gz')



# Definition of functions


def specs_prediction(specs):
    y_reg_predict = model.predict(specs,verbose=0)
    y_reg_predict = y_scaler.inverse_transform(y_reg_predict)
    y_reg_predict = pd.DataFrame(y_reg_predict,columns=dv_name)
    
    
    
    specs_val = df_val[['SNR', 'OSR', 'Power']]
    df_predict = pd.concat([specs_val.reset_index(drop=True),y_reg_predict.reset_index(drop=True)],axis=1)
    return df_predict    


def random_factor(value, range_val=0.05):
    alpha = random.uniform(-range_val, range_val)
    return value * (1 + alpha)
    
random_factor_vec = np.vectorize(random_factor)


# Validation in SIMSIDES
n = df_val.shape[0]

dv_name = df_val.columns.tolist()
for name in ['SNR', 'OSR', 'Power']:
    dv_name.remove(name)

specs = df_val[['SNR', 'OSR', 'Power']].values

num_iterations = 30
print('Making predictions...')

specs_var = np.zeros_like(specs)
specs_var[:,1] = specs[:,1]
for i in tqdm(range(num_iterations)):
    range = 0.10
    if i==0:
        range=0
    specs_var[:,0] =  random_factor_vec(specs[:,0],range_val=range)  
    specs_var[:,2] =  random_factor_vec(specs[:,2],range_val=range)  
    df_predict = specs_prediction(specs_var)
    df_predict.to_csv(f'VALIDATION/VAL-DS/Multiple-Iterations-SC/SCANN_{model_name}_val_{i+1}.csv',index=False)


