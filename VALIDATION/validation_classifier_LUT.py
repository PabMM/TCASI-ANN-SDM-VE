import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from pandas import read_csv
from tensorflow import keras
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm
import pickle
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def look_at_table(df,look):
    df_train,df_val = train_test_split(df,test_size=0.2, random_state=1)
    
    # Assume X is your original dataset and rows_to_compare are the rows to compare in X
    X_train = df_train[['SNR','OSR','Power']].values
    Y_train = df_train.drop(['SNR', 'OSR','Power'], axis=1).values

    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    look_norm = scaler.transform(look)

    # Calculating the Manhattan distance between the normalized rows and rows to compare
    distances = manhattan_distances(X_train_norm, look_norm)

    # Index of the row in X with the minimum distance to each row to compare
    closest_row_indices = distances.argmin(axis=0)

    # The rows in X that are closest to the rows to compare

    closest_rows = Y_train[closest_row_indices]
    return closest_rows

def random_factor(value, range_val=0.05):
    alpha = random.uniform(-range_val, range_val)
    return value * (1 + alpha)
    
random_factor_vec = np.vectorize(random_factor)

def validation_SC(df_val,model_name,classifier_model = ''):

    # load dataset
    
    file_name = 'DATA-SETS/data_'+model_name+'.csv'

    df = read_csv(file_name)
    dv_name = df.columns.tolist()
    for name in ['SNR', 'OSR', 'Power']:
        dv_name.remove(name)



    num_iterations = 10
    print('Making predictions...')
    specs_val = df_val[['SNR', 'OSR', 'Power']].values
    y_predict = look_at_table(df,specs_val)
    
    for i in tqdm(range(num_iterations)):
        range_val = 0.05
        if i==0:
            range_val=0
         
        y_predict_var = random_factor_vec(y_predict,range_val=range_val)
        y_predict_var = pd.DataFrame(y_predict_var,columns=dv_name)

        specs_val = df_val[['SNR', 'OSR', 'Power']]
        df_predict = pd.concat([specs_val.reset_index(drop=True),y_predict_var.reset_index(drop=True)],axis=1)
        df_predict.to_csv(f'VALIDATION/VAL-DS/Multiple-Iterations-LUT/classifier{classifier_model}_{model_name}_val_{i+1}.csv',index=False)


# validation data set
classifier_name = 'classifier'
file_name = 'DATA-SETS/data_'+classifier_name

df_val = read_csv(file_name+'_cross_val.csv')

# encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('CLASSIFIER/model/'+classifier_name+'_classes.npy',allow_pickle=True)

model = 'GB' # '' for ANN




print(f'Classifier Model {model}')
classifier_scaler = joblib.load('CLASSIFIER/model/classifier_scaler.gz')
X_val = df_val[['SNR','OSR','Power']]
column_names = X_val.columns.to_list()
scaled_values = pd.DataFrame(classifier_scaler.transform(X_val),columns=column_names)

classifier = pickle.load(open(f'CLASSIFIER/model/{model}_model.sav', 'rb'))
y_class_predict = classifier.predict(scaled_values)
# Divide df_val into diffirent sub df by y_class_predict

dfs = [df_val[y_class_predict==model_name] for model_name in encoder.classes_]

# Make predictions

for df_val,model_name in zip(dfs,encoder.classes_):
    print(model_name)
    validation_SC(df_val,model_name,classifier_model=model)