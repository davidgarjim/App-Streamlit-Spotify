#**MAIN APP**

#pip install -r requirements.txt
import streamlit as st
import pandas as pd
from def_app import (importar_csv, archivo_a_utilizar, genre, artistas, albums, playlist,
                     llevarlo_a_spotify, clean_df, conocer_variables, columns_var)
from io import StringIO
import seaborn as sns
from matplotlib import pyplot as plt

def main():

    st.title("📊 BIENVENIDO A ANALIZANDO MIS FAVORITAS DE SPOTIFY 📊")
    st.sidebar.header("Opciones")
    default_file_path = 'favoritas_hasta_septiembre24.csv'

    nombre = st.text_input("¿Cómo te llamas?")

    if nombre:
        st.success(f"¡Hola {nombre}! Paso a mostrarte el proceso que he seguido.")

        uploaded_file = importar_csv()  # Llama a la función para importar el archivo CSV

        df = archivo_a_utilizar(None, uploaded_file, default_file_path)

        if df is None or df.empty:
            st.error("No se pudo cargar el archivo CSV. Por favor, verifica el archivo.")
            return  # Salir si hay un error

        st.write("DataFrame cargado:", df.shape)


        visulizaciones = st.sidebar.multiselect(
            "¿Qué quieres ver?",
            ["Ver el DF en bruto", "Ver el DF limpio", "Explicación de las variables",
             "EDA y Análisis de las Variables Numéricas:",
             "Comparación con las más escuchadas de la historia y de 2023",
             "Géneros, Artistas, Canciones y Álbums", "Playlists automáticas con IA"]
        )
        df_bruto = df
        df = clean_df(df)

        if visulizaciones:
            for visualizacion in visulizaciones:
                if visualizacion == "Ver el DF en bruto":
                    st.subheader("\n\nDataframe en bruto")
                    st.dataframe(df_bruto)

                elif visualizacion == "Ver el DF limpio":
                    try:
                        st.subheader("\n\n\nDataframe limpio")
                        st.dataframe(df)
                        st.write("""\nEn este caso he renombrado ciertas columnas, he pasado los milisegundos a segundos y
                                he eliminado las canciones repetidas (Muchos artistas sacan primero un singles y después introducen
                                estas canciones en un albúm. A veces Spotify también falla), aunque en este caso no había duplicadas exactas.

                                \n\n En este apartado me gusta ver las canciones más antiguas, más nuevas, las más vivas,
                                cuáles han sido grabadas en directo, las más populares, etc.
                                """)
                    except ValueError as ve:
                        st.error(f"Error en la limpieza del DataFrame: {ve}")
                    except Exception as e:
                        st.error(f"Ocurrió un error al limpiar el DataFrame: {e}")

                elif visualizacion == "Explicación de las variables":
                    st.subheader('\n\nExplicación de las variables:')
                    explicacion = conocer_variables()
                    st.write(explicacion)

                elif visualizacion == "EDA y Análisis de las Variables Numéricas:":
                    st.markdown('\n\n ## **EDA y Análisis de las Variables Numéricas:**')

                    with st.expander('Ver nombre de las columnas'):

                        st.markdown("\n\n ### Columnas y su información:")
                        st.write(", ".join(df_bruto.columns))
                        st.markdown("\n\n ### Información del DataFrame:")
                        st.write("\n(Aplicamos un .info)")
                        buffer = StringIO()
                        df_bruto.info(buf=buffer)
                        s = buffer.getvalue()
                        st.text(s)

                    with st.expander('Ver duplicados'):
                        duplicados = df_bruto['ID'].duplicated().sum()
                        #duplicados_2 = df_bruto[['Track Name', 'Artist']].duplicated().sum()
                        st.markdown(f"\n\n ### Número de ID duplicados: {duplicados}")


                    with st.expander('Ver la estadísticas descriptivas'):
                        st.markdown("\n\n ### Estadísticas descriptivas:")
                        st.dataframe(df.describe())


                    with st.expander('Ver las gráficas'):
                        st.markdown("\n\n ### Gráficos:")
                        tipos_de_graficos = ['Gráfico de distorsión', 'Histograma', 'Correlaciones: Mapa de calor']
                        grafico = st.selectbox("¿Qué gráfico quieres ver?", tipos_de_graficos)


                        if grafico == 'Gráfico de distorsión':
                            x_axis = st.selectbox("Selecciona la variable para el eje X:", columns_var)
                            y_axis = st.selectbox("Selecciona la variable para el eje Y:", columns_var)
                            selected_color = st.color_picker("Elige un color para el gráfico", "#0083F9")

                            if st.button("Mostrar gráfico"):
                                plt.figure(figsize=(10, 6))
                                plt.scatter(df[x_axis], df[y_axis], color=selected_color)
                                plt.title(f'Gráfico de distorsión: {y_axis} vs {x_axis}')
                                plt.xlabel(x_axis)
                                plt.ylabel(y_axis)
                                plt.grid()
                                st.pyplot(plt)

                        if grafico == 'Histograma':

                            selected_vars = st.multiselect("Selecciona las variables para los histogramas:", columns_var)
                            selected_color = st.color_picker("Elige un color para el histograma",
                                                             "#0083F9")


                            if st.button("Mostrar histogramas"):
                                if len(selected_vars) == 0:
                                    st.warning("Por favor, selecciona al menos una variable para crear un histograma.")
                                else:
                                    for var in selected_vars:
                                        plt.figure(figsize=(10, 6))
                                        plt.hist(df[var], bins=10, color=selected_color, alpha=0.7)  # Histograma
                                        plt.title(f'Histograma de {var}')
                                        plt.xlabel(var)
                                        plt.ylabel(f"Frecuencia de {var}")

                                        plt.grid()
                                        st.pyplot(plt)

                        if grafico == 'Correlaciones: Mapa de calor':
                            correlation_matrix = df[columns_var].corr()


                            plt.figure(figsize=(14, 10))
                            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
                                        linewidths=.5)
                            plt.title("Mapa de Calor de Correlaciones")
                            plt.xticks(rotation=45)
                            plt.yticks(rotation=45)
                            plt.tight_layout()

                            st.pyplot(plt)


                elif visualizacion == "Comparación con las más escuchadas de la historia y de 2023":
                    st.subheader('\n\nComparación con las más escuchadas de la historia y de 2023:')

                    df_2023 = pd.read_csv('top_canciones_2023_espaa.csv')
                    df_history = pd.read_csv('top_100_most_streamed_songs_on_spotify_updated.csv')
                    clean_df(df_2023)
                    clean_df(df_history)

                    df_2023_var = df_2023[columns_var]
                    df_history_var = df_history[columns_var]
                    df_favs_var = df[columns_var]

                    df_combined = pd.concat([
                        df_favs_var.assign(Category='Favoritas'),
                        df_history_var.assign(Category='Más Streameadas Historia'),
                        df_2023_var.assign(Category='Más Escuchadas 2023')
                    ], ignore_index=True)

                    # Reemplazar valores de duración superiores a 7 minutos por 7 minutos para eliminar ruido y limitar outliers
                    df_combined.loc[df_combined['Duration'] > 420, 'Duration'] = 420

                    variable = st.selectbox("Selecciona la variable a visualizar", options=columns_var)
                    plot_type = st.selectbox("Selecciona el tipo de gráfico", options=["violin", "box", "barras"])

                    if plot_type in ["violin", "box"]:
                        plt.figure(figsize=(12, 6))
                        if plot_type == 'violin':
                            sns.violinplot(x='Category', y=variable, hue='Category', data=df_combined, palette="muted",
                                           dodge=False)
                        else:
                            sns.boxplot(x='Category', y=variable, hue='Category', data=df_combined, palette="Set2")

                        plt.title(f'Distribución de {variable} ({plot_type.capitalize()} Plot)')
                        plt.ylabel(variable)
                        plt.xlabel("Categoría")
                        st.pyplot(plt)

                    elif plot_type == "barras":
                        means = [df[variable].mean() for df in [df_2023_var, df_history_var, df_favs_var]]
                        medians = [df[variable].median() for df in [df_2023_var, df_history_var, df_favs_var]]

                        metric_type = st.selectbox("Selecciona el tipo de métrica", options=["Media", "Mediana"])
                        values = means if metric_type == "Media" else medians

                        plt.figure(figsize=(8, 6))
                        plt.bar(['Más Escuchadas 2023', 'Más Streameadas Historia', 'Favoritas'], values,
                                color=['blue', 'green', 'red'])
                        plt.title(f'{metric_type} de {variable}')
                        plt.ylabel(metric_type)

                        if variable == 'Acousticness':
                            plt.ylim(0, 0.5)
                        elif variable == 'Danceability':
                            plt.ylim(0, 1)
                        elif variable == 'Popularity':
                            plt.ylim(0, 100)

                        st.pyplot(plt)

                    st.write("\n\n\nVariables a visualizar recomendadas: Duración, Popularity y Energy.")

                elif visualizacion == "Géneros, Artistas, Canciones y Álbums":
                    with st.expander("Ver Géneros"):
                        genre(df)

                    with st.expander("Ver Artistas"):
                        artistas(df)

                    with st.expander("Ver Álbums"):
                        albums(df)

                elif visualizacion == "Playlists automáticas con IA":
                    df_clusters, modelo_clustering = playlist(df)
                    llevarlo_a_spotify(df, df_clusters, modelo_clustering)

        else:
            st.warning("Por favor, selecciona al menos una opción para visualizar.")

    else:
        st.warning("Introduce tu nombre para continuar.")
        st.stop()


if __name__ == '__main__':
    main()
