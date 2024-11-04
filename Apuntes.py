'''Instala Streamlit
pip install Streamlit
pip install scikit-learn
pip install xgboost
pip install matplotlib
pip install seaborn
streamlit run app.py
'''

# Importar librerías
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


def main():
    st.title("📊 BIENVENIDO A ANALIZANDO MIS FAVORITAS DE SPOTIFY")  # H1
    st.markdown(
        "Esta aplicación te permite explorar tus datos de Spotify de manera interactiva y entrenar modelos.")  # Descripción general

    # Cargar el dataset
    st.subheader("Cargar tu dataset")  # H2
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataframe en bruto")  # H2
        st.dataframe(df)  # Mostrar el dataframe

        # Análisis básico
        st.subheader("Análisis del Dataset")  # H2
        st.write("Descripción del dataset:")
        st.write(df.describe())  # Descripción estadística

        # Mostrar columnas
        if st.checkbox("Mostrar columnas"):
            st.write(df.columns)  # Mostrar nombres de columnas

        # Filtrar datos por columna
        st.subheader("Filtrar datos")  # H2
        column_to_filter = st.selectbox("Selecciona una columna para filtrar", df.columns)
        filter_value = st.text_input("Introduce el valor a filtrar:")
        if st.button("Filtrar"):
            filtered_df = df[df[column_to_filter].astype(str).str.contains(filter_value, na=False)]
            st.write("Dataframe filtrado:")
            st.dataframe(filtered_df)  # Mostrar dataframe filtrado

        # Visualización de datos
        st.subheader("Visualización de datos")  # H2
        if st.checkbox("Mostrar gráfico de dispersión"):
            x_axis = st.selectbox("Selecciona columna para el eje X", df.columns)
            y_axis = st.selectbox("Selecciona columna para el eje Y", df.columns)
            fig, ax = plt.subplots()
            ax.scatter(df[x_axis], df[y_axis], alpha=0.7)
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f'Gráfico de dispersión: {y_axis} vs {x_axis}')
            st.pyplot(fig)  # Mostrar gráfico de dispersión

        # Análisis de correlación
        st.subheader("Matriz de correlación")  # H2
        if st.checkbox("Mostrar matriz de correlación"):
            correlation_matrix = df.corr()
            plt.figure(figsize=(10, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title("Matriz de Correlación")
            st.pyplot()  # Mostrar matriz de correlación

        # Histogramas
        st.subheader("Histogramas de columnas numéricas")  # H2
        if st.checkbox("Mostrar histogramas"):
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            selected_numeric_col = st.selectbox("Selecciona una columna numérica", numeric_cols)
            fig, ax = plt.subplots()
            ax.hist(df[selected_numeric_col], bins=20, color='blue', alpha=0.7)
            ax.set_title(f'Histograma de {selected_numeric_col}')
            ax.set_xlabel(selected_numeric_col)
            ax.set_ylabel('Frecuencia')
            st.pyplot(fig)  # Mostrar histograma

        # Análisis de valores únicos
        st.subheader("Valores únicos por columna")  # H2
        if st.checkbox("Mostrar valores únicos"):
            unique_column = st.selectbox("Selecciona una columna", df.columns)
            unique_values = df[unique_column].unique()
            st.write(f"Valores únicos en {unique_column}: {unique_values}")  # Mostrar valores únicos

        # Análisis de agrupamiento
        st.subheader("Análisis de agrupamiento")  # H2
        if st.checkbox("Mostrar media agrupada"):
            group_by_column = st.selectbox("Selecciona columna para agrupar", df.columns)
            if st.checkbox("Seleccionar columnas para calcular la media"):
                selected_cols = st.multiselect("Selecciona columnas", df.columns)
                if selected_cols:
                    grouped_df = df.groupby(group_by_column)[selected_cols].mean()
                    st.write("Media agrupada por:", group_by_column)
                    st.dataframe(grouped_df)  # Mostrar media agrupada

        # Modelos de Machine Learning
        st.subheader("Entrenamiento de modelos de Machine Learning")  # H2
        if st.checkbox("Entrenar modelos de regresión"):
            # Seleccionar características y la variable objetivo
            target = st.selectbox("Selecciona la variable objetivo (Y)", df.columns)
            features = st.multiselect("Selecciona las características (X)", df.columns.drop(target))

            if st.button("Entrenar modelos"):
                if len(features) > 0:
                    # Dividir los datos en conjunto de entrenamiento y prueba
                    X = df[features]
                    y = df[target]
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Crear y entrenar los modelos
                    models = {
                        "Regresión Lineal": LinearRegression(),
                        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
                    }

                    # Almacenar las métricas
                    metrics = {}

                    for model_name, model in models.items():
                        model.fit(X_train, y_train)  # Entrenar el modelo
                        y_pred = model.predict(X_test)  # Realizar predicciones

                        # Calcular métricas
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        metrics[model_name] = {"MSE": mse, "R2": r2}  # Guardar MSE y R2

                    # Mostrar las métricas de cada modelo
                    st.write("Métricas de los modelos entrenados:")
                    metrics_df = pd.DataFrame(metrics).T
                    st.dataframe(metrics_df)  # Mostrar las métricas en un dataframe

                    # Mostrar coeficientes del modelo de regresión lineal
                    if "Regresión Lineal" in metrics:
                        coefficients = pd.DataFrame(models["Regresión Lineal"].coef_, features, columns=['Coeficiente'])
                        st.write("Coeficientes del modelo de Regresión Lineal:")
                        st.dataframe(coefficients)


if __name__ == '__main__':
    main()
