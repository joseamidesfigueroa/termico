import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import requests
import warnings
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

warnings.filterwarnings("ignore")

# Carga de datos desde API
fecha_ini = '1990-01-01'
fecha_fin = '2020-12-31'
estacion = 'Los Naranjos'
frecuencia = 'Y' # 'D'=Day; 'W'=Week; 'M'=Month; 'Y'=Year
df = pd.read_json(f'http://200.89.81.42:8100/api/HistoricosProcesados/{estacion.replace(" ","%20")}/{fecha_ini}/{fecha_fin}')

df['fecha'] = pd.to_datetime(df['fecha'])
df = df.set_index('fecha')

df = df.loc[fecha_ini:fecha_fin]
df = df[df['estacion'] == estacion]

indice_series = indice_series.resample(frecuencia).mean()

# Entrenar el modelo SARIMAX con los datos de temperatura y seleccionar los parámetros adecuados para el modelo
model = SARIMAX(indice_series, order=(0,0,0), seasonal_order=(0,1,0,len(indice_series)))
model_fit = model.fit()
# Generar pronósticos para 10 períodos a partir del tipo de frecuencia de tiempo dado
predictions = model_fit.predict(start=len(indice_series), end=len(indice_series)+10)

# Graficar los datos reales y los datos de pronóstico
plt.figure(figsize=(14, 10))
plt.plot(indice_series,color='blue',label='Datos Entrenamiento')
plt.plot(predictions, color='red',label='Datos Pronóstico')
plt.xlabel('Fecha')
plt.ylabel('Indice de confort')
plt.show()