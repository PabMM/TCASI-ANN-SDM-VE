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



def random_factor(value, range_val=0.05):
    alpha = random.uniform(-range_val, range_val)
    return value * (1 + alpha)
    
random_factor_vec = np.vectorize(random_factor)

def validation_SC(specs,model_name,classifier_model = ''):

    # load dataset
    
    file_name = 'DATA-SETS/data_'+model_name+'.csv'

    df = read_csv(file_name)
    dv_name = df.columns.tolist()
    for name in ['SNR', 'OSR', 'Power']:
        dv_name.remove(name)


    #load model
    model = keras.models.load_model('REGRESSION-ANN/models/'+model_name)
    

    # load scaler 
    y_scaler = joblib.load('REGRESSION-ANN/scalers/model_'+model_name+'_scaler.gz')


    # Validation in SIMSIDES
    
    

    num_iterations = 10
    print('Making predictions...')

    y_reg = model.predict(specs,verbose=0)
    y_reg = y_scaler.inverse_transform(y_reg)
    
    for i in tqdm(range(num_iterations)):
        range_val = 0.05
        if i==0:
            range_val=0
         
        y_reg_predict = random_factor_vec(y_reg,range_val=range_val)
        y_reg_predict = pd.DataFrame(y_reg_predict,columns=dv_name)

        specs_val = pd.DataFrame(specs,columns=['SNR', 'OSR', 'Power'])
        df_predict = pd.concat([specs_val.reset_index(drop=True),y_reg_predict.reset_index(drop=True)],axis=1)
        df_predict.to_csv(f'VALIDATION/VAL-DS/Multiple-Iterations-OPT/classifier{classifier_model}_{model_name}_val_{i+1}.csv',index=False)


# validation data set
classifier_name = 'classifier'
file_name = 'DATA-SETS/data_'+classifier_name


# encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('CLASSIFIER/model/'+classifier_name+'_classes.npy',allow_pickle=True)

model = 'GB' # '' for ANN


# Divide df_val into diffirent sub df by y_class_predict
print(encoder.classes_)



# Make predictions
specs = [[[140.1,64,1.76e-4]],
         [[90.15,128,0.4e-3]],
         [[88.1,128,10.8e-3]],
         [[116.2,128,4.1e-3]]]
for spec,model_name in zip(specs,encoder.classes_):
    print(model_name)
    validation_SC(spec,model_name,classifier_model=model)