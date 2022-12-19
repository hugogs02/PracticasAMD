import pandas as pd
import numpy as np
import seaborn as sb
from matplotlib import pyplot as plt
from statsmodels import api as st
sb.set()

#Importamos el csv
serie=pd.read_csv('IRRA_Santiago_limpio.csv')

#Vemos el dataframe y un resumen
print(serie.head(4))
print(serie.tail(4))
print(serie.shape)
print(serie.describe())

#Obtenemos la AFC de la serie, la primera y la segunda diferencia
IR=serie.loc[:,"IR"]
st.graphics.tsa.plot_acf(IR,title='Autocorrelación IR')

IRd1=np.diff(IR)
st.graphics.tsa.plot_acf(IRd1,title='Autocorrelación IR (primera diferencia)')

IRd2=np.diff(IRd1)
st.graphics.tsa.plot_acf(IRd2,title='Autocorrelación IR (segunda diferencia)')

#plt.show()


#Volvemos a importar el csv parseando las fechas y estableciendo el índice
serie=pd.read_csv('IRRA_Santiago_limpio.csv',index_col=0,parse_dates=['Fecha'])

#Descomponemos la serie
desc=st.tsa.seasonal_decompose(serie.IR,model='additive', period=2981)
desc.plot()

desc=st.tsa.seasonal_decompose(serie.IR,model='additive', period=186)
desc.plot()

plt.show()

#Hacemos la interpolación linear
desc2=st.tsa.seasonal_decompose(serie.IR,model='additive', period=186,extrapolate_trend='freq')
tend=desc2.trend
inter=tend.interpolate(method='linear')
inter.plot()
plt.show()
