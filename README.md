# Análizador y Generador de Playlists Automáticas con IA

Este proyecto permite a los usuarios analizar diferentes playlist y generar playlists automáticas en Spotify mediante técnicas de **clustering** y **análisis de características de canciones**. Utilizando la API de Spotify y algoritmos de machine learning, como **KMeans**, **DBSCAN**, y **Mean Shift**, esta aplicación organiza tus canciones en playlists basadas en sus características musicales.

## Ejecutar Aplicación

```bash
streamlit run app.py
```

## Características

- **Clustering de Canciones**: Agrupa automáticamente canciones en clusters en función de características como `Danceability`, `Energy`, `Tempo`, `Liveness` y `Acousticness`.
- **Análisis de Variables con SHAP**: Visualiza las variables más influyentes para la clasificación de las canciones en clusters.
- **Creación de Playlists en Spotify**: Autenticación en Spotify para crear playlists personalizadas directamente en la cuenta del usuario.

## Tecnologías Utilizadas

- **Python**: Lenguaje de programación principal.
- **Spotipy**: Librería para acceder a la API de Spotify.
- **Streamlit**: Framework para crear aplicaciones interactivas de datos.
- **Scikit-Learn**: Algoritmos de clustering y preprocesamiento de datos.
- **SHAP**: Interpretación del modelo de clustering.
- **XGBoost**: Modelo supervisado para el análisis de variables.

## Requisitos Previos

1. **Python**:
   - Python 3.7 o superior.

2. **Dependencias**:
   - Instala las dependencias necesarias ejecutando:
     ```bash
     pip install -r requirements.txt
     ```

3. **Cuenta de Spotify**:
   - Para poder utilizar la exportación de playlist debes tener tus propias credeciales de Spotify.

## Instalación

1. **Clonar el Repositorio**:
   ```bash
   git clone https://github.com/tuusuario/generador-playlists-spotify.git
   cd generador-playlists-spotify

## Estructura del Proyecto

├── app.py                   # Archivo principal de la aplicación Streamlit
├── def_app.py               # Funciones de clustering y autenticación
├── requirements.txt         # Lista de dependencias de Python
├── README.md                # Documentación del proyecto
└── data/                    # Carpeta para almacenar los datos de entrada
