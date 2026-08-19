import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Predicción TRM Colombia",
    page_icon="📈",
    layout="wide"
)

EWMA_LAMBDA = 0.94  # Parámetro estándar RiskMetrics para volatilidad diaria

# ==============================================================================
# LÓGICA DE DATOS Y MOTOR PREDICTIVO INTRA-APP
# ==============================================================================
@st.cache_data(ttl=3600)
def obtener_datos_mercado_trm(fecha_inicio="2019-01-01"):
    """Descarga la serie histórica DIARIA de la TRM oficial de Colombia."""
    url = "https://www.datos.gov.co/resource/32sa-8pi3.json"
    params = {
        "$select": "vigenciadesde, valor",
        "$where": f"vigenciadesde >= '{fecha_inicio}'",
        "$order": "vigenciadesde ASC",
        "$limit": 50000,
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not data:
            raise ValueError("Respuesta vacía de la API.")

        df = pd.DataFrame(data)
        df["fecha"] = pd.to_datetime(df["vigenciadesde"]).dt.normalize()
        df["trm"] = df["valor"].astype(float)
        df = (
            df[["fecha", "trm"]]
            .dropna()
            .drop_duplicates(subset="fecha")
            .set_index("fecha")
            .sort_index()
        )
        return df, True

    except Exception:
        # Fallback en caso de fallo de conexión/API
        fechas = pd.date_range(start=fecha_inicio, end=datetime.now(), freq="D")
        fechas = fechas[fechas.weekday < 5]
        trm_sintetica = 3900 + np.cumsum(np.random.normal(0, 8, len(fechas)))
        df = pd.DataFrame({"trm": trm_sintetica}, index=fechas)
        df.index.name = "fecha"
        return df, False


@st.cache_data(ttl=3600)
def ejecutar_proyeccion_trm(horizontes=[30, 60, 90], simulaciones=10000, seed=42):
    """
    Ejecuta el análisis predictivo cuantitativo (GBM + EWMA) 
    usando los datos históricos cargados en app.py.
    """
    if seed is not None:
        np.random.seed(seed)

    df_trm, es_real = obtener_datos_mercado_trm()
    retornos = np.log(df_trm["trm"] / df_trm["trm"].shift(1)).dropna()

    # Estimación de volatilidad mediante EWMA (RiskMetrics)
    media_diaria = retornos.mean()
    alpha = 1 - EWMA_LAMBDA
    var_ewma = retornos.ewm(alpha=alpha, adjust=False).var().iloc[-1]
    vol_diaria = np.sqrt(var_ewma)

    ultima_trm = df_trm["trm"].iloc[-1]
    max_dias = max(horizontes)

    # Simulación Estocástica de Trayectorias (Movimiento Browniano Geométrico)
    drift = media_diaria - 0.5 * (vol_diaria ** 2)
    Z = np.random.standard_normal(size=(max_dias, simulaciones))
    log_retornos = drift + vol_diaria * Z

    # Trayectorias acumuladas
    cum_log_ret = np.vstack([np.zeros((1, simulaciones)), np.cumsum(log_retornos, axis=0)])
    trayectorias = ultima_trm * np.exp(cum_log_ret)

    # Extracción de percentiles por horizonte
    proyecciones = {
        "spot_actual": ultima_trm,
        "es_dato_real": es_real,
        "horizontes": {}
    }

    for h in horizontes:
        precios_h = trayectorias[h, :]
        p5 = np.percentile(precios_h, 5)     # Mínimo (Piso)
        p50 = np.percentile(precios_h, 50)   # Esperado (Mediana)
        p95 = np.percentile(precios_h, 95)   # Máximo (Techo)

        proyecciones["horizontes"][f"{h}_dias"] = {
            "dias": h,
            "trm_min_p5": round(p5, 2),
            "trm_esperada_p50": round(p50, 2),
            "trm_max_p95": round(p95, 2)
        }

    return proyecciones, df_trm


# ==============================================================================
# INTERFAZ DE USUARIO EN STREAMLIT
# ==============================================================================
st.title("📈 Análisis Predictivo TRM Colombia")
st.caption("Carga y proyección automática con datos históricos oficiales (Datos Abiertos CO).")

# Carga automática al iniciar la app
with st.spinner("Cargando datos históricos y ejecutando modelo predictivo..."):
    prediccion, df_historico = ejecutar_proyeccion_trm(horizontes=[30, 60, 90])

trm_spot = prediccion["spot_actual"]

# Muestra estado de la fuente
if prediccion["es_dato_real"]:
    st.success("✅ Datos históricos sincronizados con la API de Datos Abiertos de Colombia.")
else:
    st.warning("⚠️ No se pudo conectar a la API en vivo. Mostrando datos proyectados de contingencia.")

st.markdown("---")

# Métrica Spot principal
st.metric(
    label="TRM Spot Actual (Último Cierre Oficial)", 
    value=f"${trm_spot:,.2f} COP"
)

st.subheader("🎯 Proyección de Escenarios (30, 60 y 90 Días)")

# Columnas para los 3 horizontes
col1, col2, col3 = st.columns(3)

mapa_horizontes = [
    ("30 Días", "30_dias", col1),
    ("60 Días", "60_dias", col2),
    ("90 Días", "90_dias", col3)
]

for titulo, clave, col in mapa_horizontes:
    h_data = prediccion["horizontes"][clave]
    variacion = h_data["trm_esperada_p50"] - trm_spot
    
    with col:
        st.markdown(f"### 📅 {titulo}")
        st.metric(
            label="Esperado (P50%)", 
            value=f"${h_data['trm_esperada_p50']:,.2f} COP",
            delta=f"{variacion:+,.2f} COP"
        )
        st.markdown(f"🔻 **Mínimo / Piso (P5%):** `${h_data['trm_min_p5']:,.2f}` COP")
        st.markdown(f"🔺 **Máximo / Techo (P95%):** `${h_data['trm_max_p95']:,.2f}` COP")

# Tabla consolidada
st.markdown("---")
st.subheader("📊 Resumen Cuantitativo de Proyección")

tabla_datos = []
for clave, h_data in prediccion["horizontes"].items():
    tabla_datos.append({
        "Horizonte": f"{h_data['dias']} Días",
        "Mínimo Proyectado (P5%)": f"${h_data['trm_min_p5']:,.2f} COP",
        "Esperado Proyectado (P50%)": f"${h_data['trm_esperada_p50']:,.2f} COP",
        "Máximo Proyectado (P95%)": f"${h_data['trm_max_p95']:,.2f} COP"
    })

df_resumen = pd.DataFrame(tabla_datos)
st.dataframe(df_resumen, use_container_width=True)

# Opción para ver el histórico de soporte de la proyección
with st.expander("🔍 Ver histórico de TRM utilizado para la calibración"):
    st.line_chart(df_historico["trm"])
