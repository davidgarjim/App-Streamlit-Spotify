import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import webbrowser


def importar_csv():
    uploaded_file = st.sidebar.file_uploader(
        "Puedes analizar también tu propia playlist (Debes descárgartelo en archivo CSV):",
        type=["csv"],
        key="csv_uploader"
    )
    st.sidebar.markdown(
        '<a href="https://www.tunemymusic.com/es/transfer/spotify-to-file" target="_blank">Puedes descargar tu propia playlist aquí</a>',
        unsafe_allow_html=True
    )
    return uploaded_file


def archivo_a_utilizar(df, uploaded_file, default_file_path):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(default_file_path)
        return df
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el archivo: {e}")
        return None

def clean_df(data):

    required_columns = ['\t\t\t\tDanceability', 'Artist Name(s)', 'Album Name', 'Track ID', 'Duration (ms)',
                        'Track Name']

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Faltan las siguientes columnas en el DataFrame: {missing_columns}")

    # Renombrar columnas
    data.rename(columns={'\t\t\t\tDanceability': 'Danceability'}, inplace=True)
    data.rename(columns={'Artist Name(s)': 'Artist'}, inplace=True)
    data.rename(columns={'Album Name': 'Album'}, inplace=True)
    data.rename(columns={'Track ID': 'ID'}, inplace=True)

    # Convertir la duración de milisegundos a segundos
    data['Duration'] = data['Duration (ms)'] / 1000
    data.drop(columns=['Duration (ms)'], inplace=True)

    data.drop_duplicates(subset=['Track Name', 'Artist'], keep='first', inplace=True)

    return data


def conocer_variables():
    explicacion = ("""Vamos a ver qué significa cada columna para poder comenzar a entender el Dataframe:

    1. Popularity (Popularidad)

        Descripción: Es una medida que indica qué tan popular es una canción en la
        plataforma de Spotify. Este valor es un número entre 0 y 100, donde 100
        representa la canción más popular. La popularidad se basa en el número de
        reproducciones recientes, cuántas veces ha sido compartida, añadida a playlists
        y otros factores que Spotify considera.
        
        Rango: 0 a 100.


    2. Duration (ms)

        Descripción: Es la duración total de la canción medida en milisegundos (ms). Puedes
        dividir este valor por 60,000 para convertirlo en minutos y obtener una métrica más
        comprensible.
        
        Ejemplo: Una duración de 240,000 ms equivaldría a una canción de 4 minutos.


    3. Danceability (Bailabilidad)

        Descripción: Esta métrica mide lo adecuado que es una canción para bailar, basada en
        varios elementos musicales como el tempo, la estabilidad rítmica, la fuerza del
        beat, y la regularidad. Un valor más alto indica que la canción es más fácil de bailar.
        
        Rango: 0 a 1. Un valor de 1 significa que la canción es extremadamente bailable.


    4. Energy (Energía)

        Descripción: Es una medida de la intensidad y actividad percibida de una canción. Las
        canciones con alta energía tienden a sentirse rápidas, fuertes y ruidosas (por ejemplo,
        música de rock o metal), mientras que las canciones con baja energía son más suaves y
        relajadas.
        
        Rango: 0 a 1, donde un valor cercano a 1 indica alta energía.


    5. Key (Tonalidad)

        Descripción: Representa la clave musical de la canción, utilizando notación musical
        estándar donde:
            0 es Do (C)
            1 es Do# (C#) o Re♭ (D♭)
            Y así sucesivamente hasta 11 (Si, o B).
            
        Rango: 0 a 11, representando cada una de las 12 notas musicales de la escala cromática.


    6. Loudness (Volumen)

        Descripción: Es una medida del volumen promedio de una canción, en decibelios (dB). Las
        canciones modernas tienden a ser más fuertes debido a la "guerra del volumen", pero el
        rango generalmente está entre -60 dB y 0 dB. Un valor más bajo indica que la canción es
        más silenciosa.
        
        Rango: -60 dB a 0 dB.


    7. Mode (Modo)

        Descripción: Indica si la canción está en un modo mayor o menor:
            1 = Modo mayor
            0 = Modo menor
        Las canciones en modo mayor suelen percibirse como más alegres o felices, mientras que
        las canciones en modo menor tienden a sonar más tristes o melancólicas.


    8. Speechiness (Locuacidad)

        Descripción: Evalúa la presencia de palabras habladas en una pista. Las canciones con altos
        valores de locuacidad suelen tener mucho contenido hablado, como podcasts o pistas de rap.
        
        Rango: 0 a 1. Valores cercanos a 1 indican que la pista es principalmente hablada (como un
        audiolibro o un discurso).


    9. Acousticness (Acústica)

        Descripción: Mide qué tan acústica es una canción, es decir, qué tan probable es que la pista
        haya sido creada con instrumentos acústicos, en lugar de electrónicos o amplificados.
        
        Rango: 0 a 1. Un valor de 1 indica una alta probabilidad de que la pista sea completamente acústica.


    10. Instrumentalness (Instrumentalidad)

        Descripción: Mide qué tan instrumental es una canción. Valores altos indican que la pista
        probablemente no contiene letras o voces cantadas. Las canciones con un valor cercano a 1 son
        casi completamente instrumentales.
        
        Rango: 0 a 1.


    11. Liveness (Vivosidad)

        Descripción: Mide la probabilidad de que una pista se haya grabado en vivo, es decir, si contiene
        elementos que denotan la presencia de un público o un entorno en vivo.
        
        Rango: 0 a 1. Un valor de 1 indica una alta presencia de componentes en vivo.


    12. Valence (Valencia)

        Descripción: Indica el nivel de positividad o negatividad emocional transmitida por una canción.
        Canciones con altos valores de valencia tienden a ser más alegres y optimistas, mientras que los
        valores bajos están asociados con emociones más tristes o sombrías.
        
        Rango: 0 a 1, donde 1 es extremadamente positivo y 0 es extremadamente negativo.


    13. Tempo

        Descripción: Mide el tempo de la canción, es decir, la velocidad o el ritmo al que se reproduce.
        Se mide en pulsos por minuto (BPM). Las canciones con un tempo más alto tienden a sentirse más
        rápidas y enérgicas.
        
        Rango: Generalmente entre 0 BPM y 250 BPM.


    14. Time Signature (Compás)

        Descripción: Representa la cantidad de tiempos por compás en una canción. Es un valor entero, y
        las firmas de tiempo comunes son 4/4, 3/4, o 5/4. En la mayoría de la música popular, el compás
        de 4 tiempos es el más común.
        
        Rango: Valores enteros como 3, 4 o 5, donde el valor más común es 4 (4 tiempos por compás).
    """)
    print(explicacion)
    return (explicacion)


columns_var = ['Popularity', 'Duration', 'Danceability', 'Energy', 'Key',
               'Loudness', 'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness',
               'Liveness', 'Valence', 'Tempo', 'Time Signature']


def genre(data):
    df_genres = data[['ID', 'Track Name', 'Genres']].set_index('ID')

    basic_music_styles = [
        'rock', 'pop', 'hip hop', 'electronic', 'jazz', 'blues',
        'metal', 'folk', 'classical', 'latin', 'reggae', 'funk',
        'soul', 'country', 'punk', 'indie', 'alternative', 'rumba',
        'flamenco', 'new wave', 'psychedelic', 'dance', 'rap',
        'trap', 'world', 'ska'
    ]

    st.write("Estilos de música seleccionados para el análisis:", ", ".join(basic_music_styles))

    def create_dummies(df, genre_column, styles):
        for style in styles:
            df[style] = df[genre_column].apply(lambda x: int(style in str(x).lower()))
        return df

    df_dummies_genres = create_dummies(df_genres.copy(), 'Genres', basic_music_styles)
    df_dummies_genres.drop('Genres', axis=1, inplace=True)

    # Combinación de géneros similares
    df_dummies_genres['hip_hop'] = df_dummies_genres[['hip hop', 'rap']].max(axis=1)
    df_dummies_genres['rumba_flamenco'] = df_dummies_genres[['rumba', 'flamenco']].max(axis=1)
    df_dummies_genres['punk_rock'] = df_dummies_genres[['punk', 'new wave']].max(axis=1)
    df_dummies_genres.drop(columns=['hip hop', 'rap', 'rumba', 'flamenco'], inplace=True)

    # Eliminar géneros ambiguos
    for genre in ['alternative', 'dance']:
        if genre in df_dummies_genres.columns:
            df_dummies_genres.drop(genre, axis=1, inplace=True)

    df_dummies_NoNames = df_dummies_genres.drop('Track Name', axis=1)

    # Ajustar columnas 'rock' y 'pop' en función de otros géneros presentes
    columns_to_check_rock = [
        'pop', 'electronic', 'jazz', 'blues', 'metal', 'folk',
        'classical', 'latin', 'reggae', 'funk', 'soul', 'country',
        'punk', 'indie', 'new wave', 'psychedelic', 'trap', 'world', 'ska',
        'hip_hop', 'rumba_flamenco', 'punk_rock'
    ]

    columns_to_check_pop = [
        'electronic', 'jazz', 'blues', 'metal', 'folk',
        'classical', 'latin', 'reggae', 'funk', 'soul', 'country',
        'punk', 'indie', 'new wave', 'psychedelic', 'trap', 'world', 'ska',
        'hip_hop', 'rumba_flamenco', 'punk_rock'
    ]

    df_dummies_NoNames['rock'] = df_dummies_NoNames.apply(
        lambda row: 0 if row[columns_to_check_rock].max() == 1 else row['rock'], axis=1
    )

    df_dummies_NoNames['pop'] = df_dummies_NoNames.apply(
        lambda row: 0 if row[columns_to_check_pop].max() == 1 else row['pop'], axis=1
    )

    df_recuento_generos = df_dummies_NoNames.sum().sort_values(ascending=False).reset_index()
    df_recuento_generos.columns = ['Género', 'Cantidad']

    st.subheader("Recuento de Géneros")
    st.dataframe(df_recuento_generos, width=1000, height=600)

    st.download_button(
        label="Descargar Recuento de Géneros",
        data=df_recuento_generos.to_csv(index=False).encode('utf-8'),
        file_name='df_recuento_generos.csv',
        mime='text/csv'
    )


def artistas(data):
    st.subheader("Top 100 artistas con más canciones favoritas")

    df_artistas_expandido = data['Artist'].str.split(',').explode()
    conteo_artistas = df_artistas_expandido.value_counts().sort_values(ascending=False)
    df_conteo_artistas = conteo_artistas.reset_index()
    df_conteo_artistas.columns = ['Artista', 'Cantidad']

    st.dataframe(df_conteo_artistas.head(100), width=1000, height=600)
    st.write(f"Total de artistas en la lista de favoritos: {conteo_artistas.count()}")

    csv_artistas = df_conteo_artistas.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar Recuento de Artistas",
        data=csv_artistas,
        file_name='df_recuento_artistas.csv',
        mime='text/csv'
    )


def albums(data):
    st.subheader("Top 50 álbums con más canciones favoritas:")

    df_recuento_albums = data['Album'].value_counts().sort_values(ascending=False).head(50)
    df_recuento_albums = df_recuento_albums.reset_index()
    df_recuento_albums.columns = ['Álbum', 'Cantidad']

    st.dataframe(df_recuento_albums, width=1000, height=600)

    csv_albums = df_recuento_albums.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar Recuento de Álbumes",
        data=csv_albums,
        file_name='df_recuento_albums.csv',
        mime='text/csv'
    )


def playlist(data):
    st.subheader("Crear Playlist basada en Clustering")

    variables_clusters = ['Danceability', 'Energy', 'Tempo', 'Liveness', 'Acousticness']
    df_clusters = data[variables_clusters + ["Track Name", "Artist", "ID"]].copy()

    # Select clustering model
    modelo_clustering = st.selectbox("Selecciona el modelo de clustering",
                                     ["KMeans", "DBSCAN", "Agglomerative Clustering", "Mean Shift",
                                      "Spectral Clustering"],
                                     key="model_selection")

    # Perform clustering and store result in `label` column
    if modelo_clustering == "KMeans":
        n_clusters = st.slider("Número de clusters para KMeans", min_value=2, max_value=20, value=8, step=1, key="kmeans_clusters")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_clusters['label'] = kmeans.fit_predict(df_clusters[variables_clusters])

    elif modelo_clustering == "DBSCAN":
        eps = st.slider("Radio máximo de distancia (eps) para DBSCAN", min_value=0.1, max_value=5.0, value=0.5,
                        step=0.1, key="dbscan_eps")
        min_samples = st.slider("Número mínimo de muestras para formar un cluster (min_samples) en DBSCAN", min_value=1,
                                max_value=20, value=5, step=1, key="dbscan_min_samples")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df_clusters['label'] = dbscan.fit_predict(df_clusters[variables_clusters])

    elif modelo_clustering == "Agglomerative Clustering":
        n_clusters = st.slider("Número de clusters para Agglomerative Clustering", min_value=2, max_value=20, value=8,
                               step=1, key="agglo_clusters")
        agglo = AgglomerativeClustering(n_clusters=n_clusters)
        df_clusters['label'] = agglo.fit_predict(df_clusters[variables_clusters])

    elif modelo_clustering == "Mean Shift":
        bandwidth = st.slider("Radio de búsqueda (bandwidth) para Mean Shift", min_value=0.1, max_value=5.0, value=1.0,
                              step=0.1, key="mean_shift_bandwidth")
        mean_shift = MeanShift(bandwidth=bandwidth)
        df_clusters['label'] = mean_shift.fit_predict(df_clusters[variables_clusters])

    elif modelo_clustering == "Spectral Clustering":
        n_clusters = st.slider("Número de clusters para Spectral Clustering", min_value=2, max_value=20, value=8,
                               step=1, key="spectral_clusters")
        spectral = SpectralClustering(n_clusters=n_clusters, assign_labels="kmeans", random_state=42)
        df_clusters['label'] = spectral.fit_predict(df_clusters[variables_clusters])

    st.write(f"Clusters generados con {modelo_clustering}:")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Energy', y='Tempo', hue='label', data=df_clusters, palette='Set1')
    plt.title(f'Clusters por Energy vs Tempo ({modelo_clustering})')
    st.pyplot(plt)

    with st.expander("Ejemplo de canciones de cada cluster"):
        st.write(f"Ejemplo de canción y artista por cluster ({modelo_clustering}):")
        for cluster in df_clusters['label'].unique():
            muestra = df_clusters[df_clusters['label'] == cluster].iloc[0]
            st.write(f"Cluster {cluster}: Canción - {muestra['Track Name']}, Artista - {muestra['Artist']}")

    with st.expander("¿Por qué estas playlist?"):
        # Interpretación con SHAP para entender la importancia de las variables en los clusters
        st.subheader("Interpretación de variables con SHAP")

        if st.button("Mostrar interpretación de variables con SHAP"):
            with st.spinner("Calculando valores SHAP... esto puede tardar bastante."):

                X = df_clusters[variables_clusters]
                y = df_clusters['label']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
                model.fit(X_train, y_train)

                explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)

                if not isinstance(X_test, pd.DataFrame):
                    X_test = pd.DataFrame(X_test, columns=variables_clusters)

                if len(shap_values.shape) > 2:
                    shap_values = shap_values[:, :, 0]

                if shap_values.shape[0] == X_test.shape[0] and shap_values.shape[1] == X_test.shape[1]:
                    st.write("Variables de importancia según SHAP:")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                    st.pyplot(fig)
                else:
                    st.error("Error en las dimensiones: `shap_values` y `X_test` deben tener el mismo número de filas y columnas.")

    return df_clusters, modelo_clustering



def abrir_autenticacion_spotify(client_id, client_secret, redirect_uri):
    auth_manager = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        scope='playlist-modify-private',
        cache_path=".spotify_cache"
    )

    # Abre el navegador para autenticar
    auth_url = auth_manager.get_authorize_url()
    webbrowser.open(auth_url)
    st.info("Autenticación en proceso. Por favor, completa la autenticación en el navegador.")


def llevarlo_a_spotify(data, df_clusters, modelo):
    label_column = 'label'
    if label_column not in df_clusters.columns:
        st.error(f"Error: La columna de etiquetas '{label_column}' no existe en el dataframe.")
        return

    # Paso 1: Autenticación en Spotify
    st.subheader("Paso 1: Autentificación en Spotify")

    client_id = st.text_input("Client ID de Spotify:")
    client_secret = st.text_input("Client Secret de Spotify:", type="password")
    redirect_uri = "https://app-spotify.streamlit.app/callback"

    # Autenticar si no está en session_state
    if st.button("Iniciar Autenticación en el Navegador"):
        if client_id and client_secret:
            abrir_autenticacion_spotify(client_id, client_secret, redirect_uri)
        else:
            st.error("Por favor, completa todos los campos de autenticación.")

    # Verificar si el usuario ya ha autenticado y cargar el cliente de Spotify
    if client_id and client_secret:
        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope='playlist-modify-private',
            cache_path=".spotify_cache"
        )

        if auth_manager.get_cached_token():
            sp = spotipy.Spotify(auth_manager=auth_manager)
            user_info = sp.current_user()
            user_id = user_info['id']
            st.success(f"Autenticado como: {user_info['display_name']}")
            st.session_state['spotify_auth'] = sp
            st.session_state['user_id'] = user_id
        else:
            st.info("Por favor, autentícate en el navegador y vuelve aquí para continuar.")

    # Paso 2: Exportación de Playlist si autenticado
    if 'spotify_auth' in st.session_state:
        st.subheader("Paso 2: Exportar Playlist a Spotify")

        nombre_playlist = st.text_input("Nombre de la Playlist:")
        cluster_id = st.selectbox("Selecciona el cluster para exportar", df_clusters[label_column].unique())
        canciones_cluster = df_clusters[df_clusters[label_column] == cluster_id]
        lista_ids = canciones_cluster['ID'].tolist()

        if not lista_ids:
            st.warning("No hay canciones en este cluster para exportar.")
            return

        if st.button("Crear Playlist en Spotify"):
            with st.spinner("Creando la playlist en Spotify y agregando canciones..."):
                try:
                    playlist = st.session_state['spotify_auth'].user_playlist_create(
                        user=st.session_state['user_id'], name=nombre_playlist, public=False)
                    st.info(f'Playlist creada con éxito: "{nombre_playlist}"')

                    st.session_state['spotify_auth'].user_playlist_add_tracks(
                        user=st.session_state['user_id'], playlist_id=playlist['id'], tracks=lista_ids)
                    st.success(f'Playlist "{nombre_playlist}" creada y canciones añadidas con éxito.')

                except Exception as e:
                    st.error("Hubo un problema al crear la playlist o añadir canciones.")
                    st.error(f"Detalles: {e}")
    else:
        st.info("Autenticación requerida. Por favor, autentícate en el navegador primero.")
