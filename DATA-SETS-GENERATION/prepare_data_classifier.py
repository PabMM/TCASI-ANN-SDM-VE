import pandas as pd
import random
from sklearn.model_selection import train_test_split
ds_name = 'classifier'

# Warning: this code will overwrite any existing data set. Make sure you want to overwrite or use other 
# ds_name insted

model_names = [ '2orSCSDM',
                '211CascadeSDM',
                '3orCascadeSDM',
                '2orGMSDM']

data_path = 'DATA-SETS/data_'

file_names = [data_path+model_name+'.csv' for model_name in model_names]

data_frames = [pd.read_csv(file_name)[['SNR','OSR','Power']] for file_name in file_names]



for df, model_name in zip(data_frames, model_names):
    df['category'] = model_name


# Find the length of the smallest DataFrame
min_length = min(len(df) for df in data_frames)

# Check if there are DataFrames with different lengths
if any(len(df) != min_length for df in data_frames):
    print("Warning: The DataFrames have different lengths")

# Randomly select representatives from each DataFrame
selected_dataframes = [df.sample(n=min_length, random_state=1) for df in data_frames]

train_dataframes = []
validation_dataframes = []
total_dataframes = []
cross_validation_dataframes =[]

for selected_df in selected_dataframes:
    train_df, validation_df = train_test_split(selected_df, test_size=0.2, random_state=1)
    train_dataframes.append(train_df)
    validation_dataframes.append(validation_df)
    total_dataframes.append(selected_df)
    cross_validation_dataframes.append(validation_df.sample(n=1000,random_state=1))



# Concatenate the selected DataFrames
df_train = pd.concat(train_dataframes, ignore_index=True)
df_val = pd.concat(validation_dataframes, ignore_index=True)
df_total = pd.concat(total_dataframes,ignore_index=True)
df_cross_val = pd.concat(cross_validation_dataframes,ignore_index=True)
# Save files

df_train.to_csv(data_path+ds_name+'_train.csv', index=False)
df_val.to_csv(data_path+ds_name+'_val.csv', index=False)
df_total.to_csv(data_path+ds_name+'_total.csv', index=False)
df_cross_val.to_csv(data_path+ds_name+'_cross_val.csv', index=False)

def test_same_representatives(dataframes, merged_df):
    # Obtener la lista única de valores de la columna 'category' de los DataFrames originales
    unique_categories = [df['category'].unique()[0] for df in dataframes]

    # Verificar que cada valor único de 'category' esté presente en la columna 'category' del DataFrame resultante
    for category in unique_categories:
        assert category in merged_df['category'].unique(), f"Representante de la categoría {category} no encontrado en el DataFrame resultante"

    print("El test pasó con éxito. Hay un mismo representante de cada DataFrame en el DataFrame resultante.")

# Supongamos que tienes una lista de DataFrames llamada 'dataframes' y 'merged_df' es el DataFrame resultante después de la concatenación
test_same_representatives(data_frames, df_train)
test_same_representatives(data_frames, df_val)
test_same_representatives(data_frames, df_total)
test_same_representatives(data_frames, df_cross_val)