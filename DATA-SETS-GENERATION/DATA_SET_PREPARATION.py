import pandas as pd
import random

# file names
file_name1 = 'DATA-SETS/data_2orSCSDM.csv'
category1  = 'SDM_2or_SC'

file_name2 = 'DATA-SETS/data_inter.csv'
category2  = 'SDM_2or_Gm'

new_file_name = 'DATA-SETS/data_MCANN.csv'

# read csv files
df1 = pd.read_csv(file_name1)
df2 = pd.read_csv(file_name2)


# Determinar el número de filas de cada DataFrame
num_rows_1 = df1.shape[0]
num_rows_2 = 16334


numeros_aleatorios = [round(random.random() * num_rows_1) for i in range(num_rows_2)]
df1 = df1.loc[numeros_aleatorios,:]

# check if the dataframes have a category
def add_category_column(df, category_str):
    if 'category' in df.columns:
        # if 'category' column already exists, return unmodified DataFrame
        return df
    else:
        # create a new 'category' column with given string value and return modified DataFrame
        df['category'] = category_str
        return df
df1 = add_category_column(df1,category1)
df2 = add_category_column(df2,category2)

# merge dataframes on 'SNR' and 'FOM' columns
df = pd.merge(df1, df2, on=['SNR', 'OSR','Power','category'], how='outer')
df = df.fillna(0)
# save result to a new csv file
df.to_csv(new_file_name, index=False)

print(df.shape)
print(df1.shape)
print(df2.shape)



