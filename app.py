import streamlit as st
import pandas as pd
# Importamos la función de proyecciones directamente
from proyecciones import calcular_proyecciones

st.title("Panel de Control y Proyecciones")

# Usamos el decorador de caché de Streamlit para no recalcular innecesariamente
@st.cache_data(show_spinner=False)
def obtener_datos_proyeccion(params):
    return calcular_proyecciones(params)

# Controles en la barra lateral o pantalla principal
st.sidebar.header("Parámetros de Entrada")
param_1 = st.sidebar.slider("Tasa de crecimiento (%)", 1, 50, 10)

# Indicador visual mientras se ejecuta internamente
with st.spinner('Ejecutando proyecciones en segundo plano...'):
    params = {'tasa': param_1}
    # Ejecución interna de las proyecciones
    df_resultado = obtener_datos_proyeccion(params)

st.success("¡Proyecciones calculadas exitosamente!")

# Visualización del resultado dentro de la app
st.subheader("Resultados de la Proyección")
st.dataframe(df_resultado, use_container_width=True)

st.line_chart(df_resultado.set_index('Fecha'))
