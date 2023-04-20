import base64
import io
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


app = Flask(__name__)

# Enlaza los archivos CSS y JS de Bootstrap
app = Flask(__name__, static_url_path='/static')

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/redesn")
def redesn():
    return render_template("redesn.html") 

@app.route("/historicog")
def historicog():
    return render_template("historicog.html") 

@app.route("/closterizacion")
def closterizacion():
    return render_template("closterizacion.html")   

@app.route("/arima")
def arima():
    return render_template("arima.html")  

@app.route("/modelo3")
def modelo3():
    return render_template("modelo3.html")  

# Página de historicos

@app.route('/historico', methods=['GET', 'POST'])

def historico():

    if request.method == 'POST':

        # Obtener los datos del formulario

        fecha_inicio = request.form['fecha_inicio']

        fecha_fin = request.form['fecha_fin']

        estacion = request.form['estacion']



        # Consultar la API con los parámetros de búsqueda

        url = f'http://200.89.81.42:8100/api/HistoricosProcesados/{estacion.replace(" ","%20")}/{fecha_inicio}/{fecha_fin}'

        response = requests.get(url)
        
	
            

        # Convertir la respuesta a JSON y mostrarla en la plantilla

        data = response.json()
        
        print(data)

        return render_template('redesn.html', data=data)

         

    # Si no se envió el formulario, mostrar el formulario

    return render_template('home.html')


        # Página de graficos

@app.route('/grafica', methods=['GET', 'POST'])

def grafica():

    if request.method == 'POST':

        # Obtener los datos del formulario

        fecha_inicio = request.form['fecha_inicio']

        fecha_fin = request.form['fecha_fin']

        estacion = request.form['estacion']
    
        # Cargar datos de la API

        df = pd.read_json(f'http://200.89.81.42:8100/api/HistoricosProcesados/{estacion.replace(" ","%20")}/{fecha_inicio}/{fecha_fin}')

        df["fecha"]= pd.to_datetime(df["fecha"]) # convertir fecha a tipo fecha
        # establecer índice en columna fecha
        cols_plot = ["fecha","temperatura","humedad","indice"]
        df.set_index("fecha", inplace=True)

        # dividir datos en train y test
        #train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
        train_df = df
        test_df = df

        # escalar datos
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df[["temperatura","humedad","indice"]])
        test_scaled = scaler.transform(test_df[["temperatura","humedad","indice"]])

        # crear y entrenar red neuronal
        model = Sequential()
        model.add(Dense(12, input_dim=3, activation='relu'))
        model.add(Dense(3))
        model.compile(loss='mse', optimizer='adam')
        model.fit(train_scaled, train_scaled, epochs=100, batch_size=1, verbose=0)

        # hacer pronósticos
        #train_predict = model.predict(train_scaled)
        test_predict = model.predict(test_scaled)

        # invertir escalamiento para graficar
        #train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        
        # Crear la figura
        fig = plt.figure(figsize=(12, 8))
        plt.plot(train_df[cols_plot[1]], label='Temperatura real', color='green')
        plt.plot(train_df.index, test_predict[:,0], label='Temperatura predicción', linestyle='--', color='gray')
        plt.legend()
        plt.xlabel("Fecha")
        plt.ylabel("Temperatura")
        plt.title(f'Temperatura Real vs Pronostico de la estación {estacion} ({fecha_inicio} - {fecha_fin})')

        # Convertir la figura en una cadena de bytes
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)

        # Codificar la cadena de bytes en formato base64
        plot_data = base64.b64encode(output.getvalue()).decode()

        # Enviar la imagen al navegador del usuario
        return render_template('historicog.html', plot_data=plot_data)   
        
        
        # Página closterización

@app.route('/graficas', methods=['GET', 'POST'])

def graficas():

    if request.method == 'POST':

        # Obtener los datos del formulario

        fecha_inicio = request.form['fecha_inicio']

        fecha_fin = request.form['fecha_fin']

        
    
        # Cargar datos de la API

        df = pd.read_json(f'http://200.89.81.42:8100/api/HistoricosProcesados/{fecha_inicio}/{fecha_fin}')
        df["fecha"]= pd.to_datetime(df["fecha"]) # convertir fecha a tipo fecha
        df.set_index("fecha", inplace = True) # establecer fecha como indice

        le_estacion = LabelEncoder()
        le_indice = LabelEncoder()

        df['estacion_encoded'] = le_estacion.fit_transform(df['estacion'])
        df['confort_encoded'] = le_indice.fit_transform(df['confort'])

        # Selección de variables para clustering
        X = df[['estacion_encoded', 'confort_encoded']]

        # Inicializar lista vacía para guardar las inercias
        inercias = []

        # Probar diferentes valores de k
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            inercias.append(kmeans.inertia_)

        # determinar el numero optimo de clusters utilizando la tecnica del codo
        n_clusters = range(1, 11)
        slope = [i - inercias[i - 1] for i in range(1, len(inercias))]
        opt_n_clusters = slope.index(max(slope)) + 1

        #print(f'El número óptimo de clusters es: {opt_n_clusters}')
        print(opt_n_clusters)

        # Aplicación de k-means con número óptimo de clusters
        kmeans = KMeans(n_clusters=opt_n_clusters)
        kmeans.fit(X)
        df['cluster'] = kmeans.predict(X)

        # Recuperar los valores originales en el eje X y eje Y
        estaciones_originales = le_estacion.inverse_transform(df['estacion_encoded'])
        indices_originales = le_indice.inverse_transform(df['confort_encoded'])

        df["estacion"] = estaciones_originales
        df["confort"] = indices_originales

        # Visualización de resultados
        plt.figure(figsize=(10,8))
        data = df.pivot_table(values='cluster', index='estacion', columns='confort')
        sns.heatmap(df.pivot_table(values='cluster',index='estacion',columns='confort'), cmap='Reds')
        plt.title(f'Clusters por Estaciones e Indice de Confort Térmico {fecha_inicio} a {fecha_fin}')
        plt.xlabel('Indice de Confort Térmico')
        plt.ylabel('Estacion')

        # Guardar la imagen en un objeto BytesIO en memoria en lugar de mostrarla
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)

        # Codificar el objeto BytesIO en base64 y enviarlo al navegador web
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('closterizacion.html', plot_url=plot_url)

        # Página SARIMAX
        
@app.route('/graficaari', methods=['GET', 'POST'])
def graficaari():
    plot_url = None
    if request.method == 'POST':
        # Obtener los datos del formulario
        fecha_inicio = request.form['fecha_inicio']
        fecha_fin = request.form['fecha_fin']
        estacion = request.form['estacion']
        frecuencia = 'Y' # 'D'=Day; 'W'=Week; 'M'=Month; 'Y'=Year

        # Consultar la API con los parámetros de búsqueda
        df = pd.read_json(f'http://200.89.81.42:8100/api/HistoricosProcesados/{estacion.replace(" ","%20")}/{fecha_inicio}/{fecha_fin}')
        
        # Convertir 'fecha' en índice y eliminar columna 'estacion'
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.set_index('fecha')
        df = df.drop(columns=['estacion'])

        # Resample de los datos a la frecuencia deseada
        df = df.loc[:, ['indice']]
        df = df.resample(frecuencia).mean()

        # Entrenar el modelo SARIMAX con los datos de temperatura y seleccionar los parámetros adecuados para el modelo
        model = SARIMAX(df, order=(0,0,0), seasonal_order=(0,1,0,len(df)))
        model_fit = model.fit()

        # Generar pronósticos para 5 períodos a partir del tipo de frecuencia de tiempo dado
        predictions = model_fit.forecast(steps=5)

        # Crear la gráfica y guardarla en memoria
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.plot(df, color='blue', label='Datos Entrenamiento')
        ax.plot(predictions, color='red', label='Datos Pronóstico')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Indice de confort')
        ax.legend()
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)

        # Codificar la imagen en base64 y mostrarla en la página HTML
        plot_url = base64.b64encode(buffer.getvalue()).decode()

    return render_template('arima.html', plot_url=plot_url)


        # Graficar los datos reales y los datos de pronóstico
        #plt.figure(figsize=(14, 10))
        #plt.plot(df, color='blue', label='Datos Entrenamiento')
        #plt.plot(predictions, color='red', label='Datos Pronóstico')
        #plt.xlabel('Fecha')
        #plt.ylabel('Indice de confort')
        #plt.show()
      

if __name__ == '__main__':
     
    app.run(host="0.0.0.0", port=8080)
