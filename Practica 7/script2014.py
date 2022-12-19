import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

# Realizamos la conexión a la base de datos y comprobamos que se haya hecho correctamente
conn_str = "host='localhost' dbname='P7_AMD' port=5432 user='postgres' password='291102'"
conn = psycopg2.connect(conn_str)
if conn == None:
    raise Exception("No se pudo conectar a la base de datos.\n")

# Guardamos los datos en un dataframe
df = pd.read_sql_query('SELECT * FROM datos2 WHERE time=2014', con=conn)

print(df.shape)

# Cerramos la conexión a la base de datos
conn.close()

SEMILLA = np.random.seed(123456789)

# Quitamos los valores vacíos
df.dropna(inplace=True)

# Quitamos columnas inútiles, así como el año y el nombre del país (para que no de problemas al normalizar)
data = df.drop(['time', 'time code', 'country name', 'country code'], axis=1)

print(data.shape)

# Normalizamos los datos
df_norm = (data-data.min())/(data.max()-data.min())

# Representamos los datos en un pairplot
sns.pairplot(data=df_norm)
plt.show()

# Obtenemos los parámetros eps y minPts
minimoPts = 2

ngbrs = NearestNeighbors(n_neighbors=minimoPts).fit(df_norm)
distances, indices = ngbrs.kneighbors(df_norm)

distances = np.sort(distances, axis=0)
distances = distances[:, (minimoPts-1)]
plt.plot(distances)
plt.axhline(y=0.22, linewidth=1, linestyle='dashed', color='k')
plt.show()

epsilon = 0.22

# Aplicamos el algoritmo DBSCAN y graficamos el resultado
dbscan = DBSCAN(eps=epsilon, min_samples=minimoPts).fit(df_norm)

dblabels = dbscan.labels_
df_norm["label"] = dblabels

cols = len(np.unique(np.array(dblabels)))

sns.pairplot(data=df_norm, hue="label", palette=sns.color_palette("husl", cols))
plt.show()

# Obtenemos la cantidad de puntos por clúster
cant = pd.DataFrame()
cant['cantidad'] = df_norm.groupby('label').size()
print(cant)


df_merged = pd.concat([df, df_norm], axis=1, join='inner')
df_merged = df_merged[['country name', 'country code', 'label']]
print(df_merged)
df_merged.to_csv('datamergedlabels2014.csv', encoding='utf-8', index=False, sep=';')

