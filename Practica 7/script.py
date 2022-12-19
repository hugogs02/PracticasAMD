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
df = pd.read_sql_query('SELECT * FROM datos', con=conn)

print(df.shape)
print(df)

# Cerramos la conexión a la base de datos
conn.close()

SEMILLA = np.random.seed(123456789)

# Normalizamos los datos
df_norm = (df-df.min())/(df.max()-df.min())

df_norm=df_norm.sample(n=200000, random_state = SEMILLA)

sns.pairplot(df_norm)
plt.show()

# Obtenemos los parámetros eps y minPts
minimoPts = 2*df_norm.shape[1]

ngbrs = NearestNeighbors(n_neighbors=minimoPts).fit(df_norm)
distances, indices = ngbrs.kneighbors(df_norm)

distances = np.sort(distances, axis=0)
distances = distances[:, (minimoPts-1)]
plt.plot(distances)
plt.axhline(y=0.10, linewidth=1, linestyle='dashed', color='k')
plt.show()

epsilon=0.1

mnpts=[10, 20]


# Aplicamos el algoritmo DBSCAN
for minimoPts in mnpts:
    print("Probando con minpts=", minimoPts)

    dbscan = DBSCAN(eps=epsilon, min_samples=minimoPts).fit(df_norm)

    dblabels = dbscan.labels_
    df_norm["label"] = dblabels

    print(dblabels)

    cols = len(np.unique(np.array(dblabels)))
    print(cols)

    sns.pairplot(data=df_norm, hue="label", palette=sns.color_palette("husl", cols))
    plt.show()


    cant = pd.DataFrame()
    cant['cantidad'] = df_norm.groupby('label').size()
    print(cant)