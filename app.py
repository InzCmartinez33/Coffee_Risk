import streamlit as st
import pandas as pd
# Importamos la función de proyección directamente de tu motor
from motor import generar_proyeccion_trm_interna, obtener_trm_hoy

st.set_page_config(page_title="Predicción TRM - Dashboard", layout="wide")

st.title("📈 Análisis Predictivo TRM Colombia")
st.markdown("Proyección estocástica basada en Movimiento Browniano Geométrico (GBM) y volatilidad EWMA.")

# ==============================================================================
# CARGA AUTOMÁTICA Y CACHÉ DEL MOTOR PREDICTIVO
# ==============================================================================
@st.cache_data(ttl=3600)  # Recalcula automáticamente cada hora
def cargar_proyecciones_automaticas():
    # Llama al motor predictivo para horizontes de 30, 60 y 90 días
    return generar_proyeccion_trm_interna(horizontes=[30, 60, 90], simulaciones=10000)

with st.spinner("Ejecutando simulaciones cuantitativas de TRM..."):
    datos_pred = cargar_proyecciones_automaticas()

trm_spot = datos_pred["spot_actual"]

# ==============================================================================
# PRESENTACIÓN DE MÉTRICAS CLAVE
# ==============================================================================
st.metric(label="TRM Spot Actual (Oficial)", value=f"${trm_spot:,.2f} COP")

st.subheader("🎯 Proyección de Escenarios (30, 60 y 90 días)")

# Creamos 3 columnas para visualización clara de horizontes
col1, col2, col3 = st.columns(3)

horizontes_map = [
    ("30 Días", "30_dias", col1),
    ("60 Días", "60_dias", col2),
    ("90 Días", "90_dias", col3)
]

for titulo, clave, col in horizontes_map:
    h_data = datos_pred["horizontes"][clave]
    with col:
        st.markdown(f"### 📅 {titulo}")
        
        # Valor Esperado (P50)
        st.metric(
            label="Esperado (P50%)", 
            value=f"${h_data['trm_esperada_p50']:,.2f}",
            delta=f"{h_data['trm_esperada_p50'] - trm_spot:,.2f} COP"
        )
        
        # Rango Estocástico Mínimo y Máximo
        st.caption(f"🔻 **Mínimo (Pish/P5%):** ${h_data['trm_min_p5']:,.2f} COP")
        st.caption(f"🔺 **Máximo (Techo/P95%):** ${h_data['trm_max_p95']:,.2f} COP")

# ==============================================================================
# TABLA RESUMEN COMPACTA
# ==============================================================================
st.markdown("---")
st.subheader("📊 Tabla Cuantitativa de Rangos")

filas = []
for clave, h_data in datos_pred["horizontes"].items():
    filas.append({
        "Horizonte": f"{h_data['dias']} Días",
        "Piso / Mínimo (P5%)": f"${h_data['trm_min_p5']:,.2f} COP",
        "Esperado / Mediana (P50%)": f"${h_data['trm_esperada_p50']:,.2f} COP",
        "Techo / Máximo (P95%)": f"${h_data['trm_max_p95']:,.2f} COP",
    })

df_resumen = pd.DataFrame(filas)
st.dataframe(df_resumen, use_container_width=True)
