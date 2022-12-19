import pandas as pd
import numpy as np
import seaborn as sns
import math
from matplotlib import pyplot as plt
from statsmodels import api as sm
import pyaf.ForecastEngine as autof
from statsmodels.tsa.statespace.sarimax import SARIMAX
sns.set()

#-----------------------------------------------------
#------------Creación de series temporales------------

#---------------------Variable TM---------------------
print("----------Variable TM------------")
# importar CSV
serieTM = pd.read_csv('./TM_Santiago_limpio.csv')

# ver dataframe
print(serieTM.head(4))
print(serieTM.tail(4))

# resumen dataframe
print(serieTM.shape)
print(serieTM.describe())

# AFC normal
TM = serieTM.loc[:,"TM"]
sm.graphics.tsa.plot_acf(TM, title="TM - Autocorrelación")

# AFC de la primera diferencia
TM_diff_1 = np.diff(TM)
sm.graphics.tsa.plot_acf(TM_diff_1, title="Primera Diferencia TM - Autocorrelación")

# AFC de la segunda diferencia
TM_diff_2 = np.diff(TM_diff_1)
sm.graphics.tsa.plot_acf(TM_diff_2, title="Segunda Diferencia TM - Autocorrelación")

#plt.show()

# importar CSV
serieTM = pd.read_csv('./TM_Santiago_limpio.csv', parse_dates = True)

# campo Fecha como índice
serieTM['Fecha'] = pd.to_datetime(serieTM['Fecha'])
serieTM = serieTM.set_index('Fecha')

# ver dataframe
print(serieTM.head(4))
print(serieTM.tail(4))

# resumen dataframe
print(serieTM.shape)
print(serieTM.describe())

# descomponer en componentes estacionales, tendencia y residuos
descomposicion_TM = sm.tsa.seasonal_decompose(serieTM.TM, model='additive', period=3001)
descomposicion_TM.plot()

#plt.show()

descomposicion_TM = sm.tsa.seasonal_decompose(serieTM.TM, model='additive', period=375)
descomposicion_TM.plot()

#plt.show()


#---------------------Variable P---------------------
print("----------Variable P------------")
#empezamos leyendo el .csv
sns.set()
serieP = pd.read_csv('P_Santiago_limpio.csv')

#ver dataframe y resumen del dataframe
print(serieP.head(4))
print(serieP.tail(4))
print(serieP.shape)
print(serieP.describe())

P = serieP.loc[:,"P"]
sm.graphics.tsa.plot_acf(P,title="P - Autocorrelacion")
plt.show()

P_diff_1 = np.diff(P)
sm.graphics.tsa.plot_acf(P_diff_1,title="Primera diferencia P - Autocorrelacion")
plt.show()

P_diff_2 = np.diff(P_diff_1)
sm.graphics.tsa.plot_acf(P_diff_2,title="Segunda diferencia P - Autocorrelacion")
plt.show()


serieP = pd.read_csv('P_Santiago_limpio.csv', parse_dates = True)
print(serieP.head(4))
print(serieP.tail(4))
print(serieP.shape)
print(serieP.describe())
serieP['Fecha'] = pd.to_datetime(serieP['Fecha'])
serieP = serieP.set_index('Fecha')

P = serieP.loc[:,"P"]

descomposicion_P = sm.tsa.seasonal_decompose(serieP.P, model= 'additive', period=2965)
descomposicion_P.plot()
plt.show()

descomposicion_P = sm.tsa.seasonal_decompose(serieP.P,model= 'additive',period=371)
descomposicion_P.plot()
plt.show()


#---------------------Variable IRRA---------------------
print("----------Variable IRRA------------")
#Importamos el csv
serieIRRA=pd.read_csv('IRRA_Santiago_limpio.csv')

#Vemos el dataframe y un resumen
print(serieIRRA.head(4))
print(serieIRRA.tail(4))
print(serieIRRA.shape)
print(serieIRRA.describe())

#Obtenemos la AFC de la serie, la primera y la segunda diferencia
IR=serieIRRA.loc[:,"Irradiacion_global_diaria"]
sm.graphics.tsa.plot_acf(IR,title='Autocorrelación IR')

IR_diff_1=np.diff(IR)
sm.graphics.tsa.plot_acf(IR_diff_1,title='Autocorrelación IR (primera diferencia)')

IR_diff_2=np.diff(IR_diff_1)
sm.graphics.tsa.plot_acf(IR_diff_2,title='Autocorrelación IR (segunda diferencia)')

plt.show()


#Volvemos a importar el csv parseando las fechas y estableciendo el índice
serieIRRA=pd.read_csv('IRRA_Santiago_limpio.csv',index_col=0,parse_dates=['Instante_lectura'])

#Descomponemos la serie
descIRRA=sm.tsa.seasonal_decompose(serieIRRA.Irradiacion_global_diaria,model='additive', period=2981)
descIRRA.plot()

descIRRA=sm.tsa.seasonal_decompose(serieIRRA.Irradiacion_global_diaria,model='additive', period=372)
descIRRA.plot()

plt.show()



#-----------------------------------------------------
#-----------------Interpolación lineal----------------

#---------------------Variable TM---------------------
print("----------Variable TM------------")
# interpolacion lineal de la tendencia
np_Tendencia = np.array(descomposicion_TM.trend)
finite_indexes = np.isfinite(np_Tendencia)
coeficientes = np.polyfit(np.array(pd.to_datetime(descomposicion_TM.trend.index).astype(int) / 10**18)[finite_indexes],np_Tendencia[finite_indexes],1)
funcion_regresion = np.poly1d(coeficientes)

# insercion de columnas tendencia y regresion en el dataframe
serieTM.insert(1, 'Tendencia', descomposicion_TM.trend.interpolate(method="linear"))
serieTM.insert(2, 'Regresion', funcion_regresion(np.array(pd.to_datetime(serieTM.index).astype(int) / 10**18)))

# gráfica: interpolacion lineal con regresion lineal
plt.plot(serieTM.index, serieTM.Tendencia, serieTM.Regresion)

# pendiente de la regresion lineal
print(coeficientes)

plt.show()


#---------------------Variable P---------------------
print("----------Variable P------------")
#Hacemos la interpolación linear de la tendencia
P_tend=np.array(descomposicion_P.trend)
indexesP=np.isfinite(P_tend)
coefs_P=np.polyfit(np.array(pd.to_datetime(descomposicion_P.trend.index).astype(int)/10**18)[indexesP],P_tend[indexesP],1)
func_P=np.poly1d(coefs_P)


#Insertamos las columnas de tendencia y regresion al dataframe
serieP.insert(1,'Tendencia',descomposicion_P.trend.interpolate(method="linear"))
serieP.insert(2,'Regresion',func_P(np.array(pd.to_datetime(serieP.index).astype(int)/10**18)))

plt.plot(serieP.index, serieP.Tendencia, serieP.Regresion)
print(coefs_P)
plt.show()


#---------------------Variable IRRA---------------------
print("----------Variable IRRA------------")
#Hacemos la interpolación linear de la tendencia
IIRA_tend=np.array(descIRRA.trend)
indexesIIRA=np.isfinite(IIRA_tend)
coefs=np.polyfit(np.array(pd.to_datetime(descIRRA.trend.index).astype(int)/10**18)[indexesIIRA],IIRA_tend[indexesIIRA],1)
func=np.poly1d(coefs)


#Insertamos las columnas de tendencia y regresion al dataframe
serieIRRA.insert(1,'Tendencia',descIRRA.trend.interpolate(method="linear"))
serieIRRA.insert(2,'Regresion',func(np.array(pd.to_datetime(serieIRRA.index).astype(int)/10**18)))

plt.plot(serieIRRA.index, serieIRRA.Tendencia, serieIRRA.Regresion)
print(coefs)
plt.show()



#-----------------------------------------------------
#----------Cálculo de la correlación cruzada----------
# CREACION DE LOS DATAFRAMES

# importar CSV
serieTM = pd.read_csv('./TM_Santiago_limpio.csv', parse_dates = True)
serieP = pd.read_csv('./P_Santiago_limpio.csv', parse_dates = True)
serieIRRA = pd.read_csv('./IRRA_Santiago_limpio.csv', parse_dates = True)

# campo Fecha como índice
serieTM['Fecha'] = pd.to_datetime(serieTM['Fecha'])
serieTM = serieTM.set_index('Fecha')

serieP['Fecha'] = pd.to_datetime(serieP['Fecha'])
serieP = serieP.set_index('Fecha')

serieIRRA['Instante_lectura'] = pd.to_datetime(serieIRRA['Instante_lectura'])
serieIRRA = serieIRRA.set_index('Instante_lectura')


# creamos dataframe conjunto
df_conjunto = serieTM
df_conjunto.insert(0, 'P', serieP.P)
df_conjunto.insert(1, 'IRRA', serieIRRA.Irradiacion_global_diaria)

# ver dataframe
print(df_conjunto.head(4))
print(df_conjunto.tail(4))
print("\n")


# CORRELACIONES CRUZADAS
# índices sin NaN
indexes_TM_P = np.isfinite(df_conjunto.TM) & np.isfinite(df_conjunto.P)
indexes_TM_IRRA = np.isfinite(df_conjunto.TM) & np.isfinite(df_conjunto.IRRA)
indexes_P_IRRA = np.isfinite(df_conjunto.P) & np.isfinite(df_conjunto.IRRA)

# cálculo de correlaciones cruzadas
ccf_TM_P = sm.tsa.stattools.ccf(df_conjunto.TM[indexes_TM_P], df_conjunto.P[indexes_TM_P])
ccf_TM_IRRA = sm.tsa.stattools.ccf(df_conjunto.TM[indexes_TM_IRRA], df_conjunto.IRRA[indexes_TM_IRRA])
ccf_P_IRRA = sm.tsa.stattools.ccf(df_conjunto.P[indexes_P_IRRA], df_conjunto.IRRA[indexes_P_IRRA])

# graficamos la correlacion cruzada
plt.plot(ccf_TM_P)
plt.show()

plt.plot(ccf_TM_IRRA)
plt.show()

plt.plot(ccf_P_IRRA)
plt.show()



#-----------------------------------------------------
#---------Filtro en tendencia con media móvil---------
#Establecemos las ventanas para la media móvil
ventana0=math.floor(375/2)
ventana1=375
ventana2=375*2

#Calculamos las medias para la primera variable, con las tres ventanas
meanTM0=df_conjunto.TM.rolling(ventana0).mean()
meanTM1=df_conjunto.TM.rolling(ventana1).mean()
meanTM2=df_conjunto.TM.rolling(ventana2).mean()

#Graficamos las medias
plt.title("Media móvil para la serie TM")
plt.plot(meanTM0, label='Ventana corta')
plt.plot(meanTM1, label='Ventana media')
plt.plot(meanTM2, label='Ventana larga')
plt.legend()
plt.show()


#Calculamos las medias para la segunda variable, con las tres ventanas
meanIRRA0=df_conjunto.IRRA.rolling(ventana0).mean()
meanIRRA1=df_conjunto.IRRA.rolling(ventana1).mean()
meanIRRA2=df_conjunto.IRRA.rolling(ventana2).mean()

#Graficamos las medias
plt.title("Media móvil para la serie IRRA")
plt.plot(meanIRRA0, label='Ventana corta')
plt.plot(meanIRRA1, label='Ventana media')
plt.plot(meanIRRA2, label='Ventana larga')
plt.legend()
plt.show()


#Calculamos las medias para la tercera variable, con las tres ventanas
meanP0=df_conjunto.P.rolling(ventana0).mean()
meanP1=df_conjunto.P.rolling(ventana1).mean()
meanP2=df_conjunto.P.rolling(ventana2).mean()

#Graficamos las medias
plt.title("Media móvil para la serie P")
plt.plot(meanP0, label='Ventana corta')
plt.plot(meanP1, label='Ventana media')
plt.plot(meanP2, label='Ventana larga')
plt.legend()
plt.show()



#-----------------------------------------------------
#----------------------Predicción----------------------
#Realizamos la predicción para la serie TM
modTM=sm.tsa.SARIMAX(df_conjunto.TM, order=(1,0,0), trend='c')
resTM=modTM.fit()
fcstTM=resTM.forecast(steps=365*2)
df_final_TM=pd.concat([df_conjunto.TM,fcstTM],ignore_index=True)

df_final_TM.iloc[0:6003].plot(color='blue',label='Datos')
df_final_TM.iloc[6004:6368].plot(color='red',label='Prediccion')
plt.gca().get_xaxis().set_visible(False)
plt.legend()
plt.show()

print(df_final_TM)
