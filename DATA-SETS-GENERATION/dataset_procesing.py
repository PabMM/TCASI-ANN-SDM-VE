import pandas as pd

# Supongamos que tienes un DataFrame llamado 'df' con tus datos
df = pd.read_csv('DATA-SETS-GENERATION/211CascadeSDM_DataSet_random.csv')

# Definimos el tamaño del muestreo (n)
n = 50000  # Puedes cambiar el valor de 'n' según tus necesidades

# Realizamos el muestreo con la función 'sample' de pandas
muestreo = df.sample(n)


# Filtrar las filas con SNR menor a 50
df['SNR'] = -df['SNR']
df_filtrado = df[df["SNR"] > 50]

# Guardar el DataFrame filtrado en un archivo CSV
df_filtrado.to_csv("datos_filtrados.csv", index=False)

print(df_filtrado)

# Si deseas guardar el nuevo DataFrame 'muestreo' en un archivo CSV, puedes hacerlo con:
df_filtrado.to_csv('DATA-SETS/data_211CascadeSDM.csv', index=False)


