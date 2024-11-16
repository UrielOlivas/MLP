import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Definición de la clase CombinedAttributesAdder
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_temp_humidity=True):
        self.add_temp_humidity = add_temp_humidity

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_temp = X.copy()
        if self.add_temp_humidity:
            X_temp['temp_humidity'] = X_temp['temp'] * X_temp['humidity']
        return X_temp

# Cargar el modelo y el pipeline
model = joblib.load("mlp (2).pkl")
pipeline = joblib.load("pipeline (2).joblib")

# Título de la aplicación
st.markdown(
    """
    <h1 style='text-align: center;'>Predicción de Casos de Dengue</h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align: center;">
        <p>Maestría en Ingeniería en Computación</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="text-align: center;">
        <p>Francisco Uriel Olivas Márquez 341948</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.image('dengue.jpg',use_container_width=True)

# Entradas del usuario
col1, col2 = st.columns(2)
tempmax = col1.number_input("Temperatura Máxima (°C)", min_value=-50.0, max_value=50.0, value=34.0)
tempmin = col1.number_input("Temperatura Mínima (°C)", min_value=-50.0, max_value=50.0, value=24.0)
temp = col1.number_input("Temperatura (°C)", min_value=-50.0, max_value=50.0, value=28.0)
feelslikemax = col1.number_input("Sensación Térmica Máxima (°C)", min_value=-50.0, max_value=50.0, value=39.0)
feelslikemin = col1.number_input("Sensación Térmica Mínima", min_value=0.0, max_value=50.0, value=25.0)
feelslike= col1.number_input("Sensación Térmica", min_value=0.0, max_value=50.0, value=32.0)
dew = col1.number_input("Rocío", min_value=0.0, max_value=50.0, value=22.0)
humidity = col1.number_input("Humedad", min_value=0.0, max_value=100.0, value=73.0)
precip = col1.number_input("Precipitación(mm)", min_value=0.0, max_value=500.0, value=3.0)
precipprob = col1.number_input("Probabilidad de Precipitación", min_value=0.0, max_value=100.0, value=44.0)
precipcover = col2.number_input("Cobertura de precipitación", min_value=0.0, max_value=500.0, value=4.0)
windspeed = col2.number_input("Velocidad del Viento", min_value=0.0, max_value=500.0, value=15.0)
winddir = col2.number_input("Dirección del Viento", min_value=0.0, max_value=500.0, value=175.0)
sealevelpressure = col2.number_input("Presión", min_value=0.0, max_value=1500.0, value=1007.0)
cloudcover = col2.number_input("Cobertura de Nubes", min_value=0.0, max_value=100.0, value=50.0)
visibility = col2.number_input("Visibilidad", min_value=0.0, max_value=50.0, value=3.75)
solarradiation = col2.number_input("Radiación Solar", min_value=0.0, max_value=500.0, value=208.0)
solarenergy = col2.number_input("Energía Solar", min_value=0.0, max_value=50.0, value=18.0)
uvindex = col2.number_input("Índice UV", min_value=0.0, max_value=50.0, value=7.0)
conditions = col2.number_input("Condiciones", min_value=0.0, max_value=4.0, value=1.19)



# Crear un DataFrame con las entradas
input_data = pd.DataFrame({
    'tempmax': [tempmax],
    'tempmin': [tempmin],
    'temp': [temp],
    'feelslikemax': [feelslikemax],
    'feelslikemin': [feelslikemin],
    'feelslike': [feelslike],
    'dew': [dew],
    'humidity': [humidity],
    'precip': [precip],
    'precipprob': [precipprob],
    'precipcover': [precipcover],
    'snow': [0],
    'snowdepth': [0],
    'windspeed': [windspeed],
    'winddir': [winddir],
    'sealevelpressure': [sealevelpressure],
    'cloudcover': [cloudcover],
    'visibility': [visibility],
    'solarradiation': [solarradiation],
    'solarenergy': [solarenergy],
    'uvindex': [uvindex],
    'conditions': [conditions]
})

# Transformar los datos utilizando el pipeline
input_data_prepared = pipeline.transform(input_data)

# Botón para hacer predicción
if st.button("Realizar Predicción"):
    prediction = model.predict(input_data_prepared)
    st.success(f"La predicción de casos es: {prediction[0]}")