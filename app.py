import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta

# =====================================================================
# CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="RiskCafe Engine - Análisis de Riesgo",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("☕ RiskCafe Engine - Gestión de Riesgo para Café")
st.markdown("Herramienta de simulación de Montecarlo y valoración de inventarios (Excelso + Pasilla) para comercializadores de café.")

# =====================================================================
# PARÁMETROS EN SIDEBAR
# =====================================================================
st.sidebar.header("⚙️ Parámetros de Operación")

USA_FACTOR_RENDIMIENTO = st.sidebar.checkbox(
    "Calcular por Factor de Rendimiento",
    value=True,
    help="Calcula las libras de café excelso y los kilos de pasilla producidos por carga de 125 kg pergamino."
)

if USA_FACTOR_RENDIMIENTO:
    FACTOR_RENDIMIENTO = st.sidebar.number_input(
        "Factor de Rendimiento (kg pergamino / 70 kg excelso)",
        value=94.0, step=0.5, format="%.2f", min_value=70.0
    )
    # Libras de excelso = (125 / Factor) * 70 kg * 2.20462 lbs/kg
    LIBRAS_POR_CARGA = (125.0 / FACTOR_RENDIMIENTO) * 70.0 * 2.20462262
    
    # Estimación de kilos de pasilla por carga de 125 kg:
    # Kilos excelso por carga = (125 / Factor) * 70
    KG_EXCELSO_POR_CARGA = (125.0 / FACTOR_RENDIMIENTO) * 70.0
    # Descontando merma/cisco promedio (~18 kg)
    KG_PASILLA_POR_CARGA = max(0.0, 125.0 - KG_EXCELSO_POR_CARGA - 18.0)
    
    st.sidebar.info(f"💡 **Excelso:** {LIBRAS_POR_CARGA:.2f} lbs ({KG_EXCELSO_POR_CARGA:.1f} kg)\n\n💡 **Pasilla estimada:** {KG_PASILLA_POR_CARGA:.1f} kg/carga")
else:
    LIBRAS_POR_CARGA = st.sidebar.number_input(
        "Libras por Carga (125 kg)",
        value=96.25, step=0.25, format="%.2f"
    )
    KG_PASILLA_POR_CARGA = st.sidebar.number_input(
        "Kilos de Pasilla por Carga (kg)",
        value=12.0, step=0.5, format="%.1f"
    )

PRECIO_PASILLA_COP_KG = st.sidebar.number_input(
    "Precio Pasilla Loca (COP/kg)",
    value=10000.0, step=500.0, format="%.0f",
    help="Precio local de venta para la pasilla por kilogramo."
)

DIFERENCIAL_CALIDAD_USD_LB = st.sidebar.number_input("Diferencial Calidad (USD/lb)", value=0.08, step=0.01)
COSTOS_EXPORTACION_USD_LB = st.sidebar.number_input("Costos Exportación (USD/lb)", value=0.12, step=0.01)
VOLUMEN_CARGAS = st.sidebar.number_input("Volumen (Cargas)", value=180, step=10, min_value=1)

TENOR_ANALISIS = st.sidebar.selectbox("Plazo de Análisis", options=["1m", "3m", "6m"], index=0)
dias_analisis = {"1m": 30, "3m": 90, "6m": 180}[TENOR_ANALISIS]

# =====================================================================
# OBTENCIÓN DE DATOS CON FALLBACK SEGURO
# =====================================================================
@st.cache_data(ttl=1800)
def cargar_datos_seguro():
    try:
        import yfinance as yf
        df_cafe = yf.Ticker("KC=F").history(period="1y")
        if not df_cafe.empty:
            precio_cafe = float(df_cafe["Close"].iloc[-1])
        else:
            precio_cafe = 245.50
    except Exception:
        precio_cafe = 245.50

    try:
        resp = requests.get("https://www.datos.gov.co/resource/32sa-8pi3.json?$limit=1&$order=vigenciadesde DESC", timeout=5)
        if resp.status_code == 200 and len(resp.json()) > 0:
            trm = float(resp.json()[0]["valor"])
        else:
            trm = 4050.0
    except Exception:
        trm = 4050.0

    return precio_cafe, trm

with st.spinner("Cargando datos de mercado..."):
    precio_ny_hoy, trm_hoy = cargar_datos_seguro()

# =====================================================================
# CÁLCULOS
# =====================================================================
def calcular_componentes_carga(precio_ny, trm, lbs_excelso, kg_pasilla, precio_pasilla_kg):
    precio_usd_lb = (precio_ny / 100) + DIFERENCIAL_CALIDAD_USD_LB - COSTOS_EXPORTACION_USD_LB
    valor_excelso_cop = precio_usd_lb * trm * lbs_excelso
    valor_pasilla_cop = kg_pasilla * precio_pasilla_kg
    precio_total_carga_cop = valor_excelso_cop + valor_pasilla_cop
    return precio_total_carga_cop, valor_excelso_cop, valor_pasilla_cop

precio_carga_hoy, valor_excelso_hoy, valor_pasilla_hoy = calcular_componentes_carga(
    precio_ny_hoy, trm_hoy, LIBRAS_POR_CARGA, KG_PASILLA_POR_CARGA, PRECIO_PASILLA_COP_KG
)

valor_portafolio_total = precio_carga_hoy * VOLUMEN_CARGAS

# =====================================================================
# VISUALIZACIÓN EN DASHBOARD
# =====================================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Café NY (KC=F)", f"{precio_ny_hoy/100:.2f} USD/lb")
col2.metric("TRM (COP)", f"${trm_hoy:,.2f}")
col3.metric("Precio Carga Ref.", f"${precio_carga_hoy:,.0f} COP")
col4.metric("Valor Portafolio Total", f"${valor_portafolio_total/1e6:,.2f} M COP")

# Desglose de Excelso vs Pasilla
st.info(
    f"📌 **Desglose de la Carga de 125 kg:** "
    f"Valor Excelso: **${valor_excelso_hoy:,.0f} COP** | "
    f"Valor Pasilla ({KG_PASILLA_POR_CARGA:.1f} kg @ ${PRECIO_PASILLA_COP_KG:,.0f}/kg): **${valor_pasilla_hoy:,.0f} COP**"
)

st.markdown("---")

st.subheader("📊 Simulación de Riesgo (Montecarlo)")

np.random.seed(42)
retornos_sim = np.random.normal(0, 0.02, (dias_analisis, 2000))

# Simulación sobre la componente de café de exportación
precios_ny_simulados = precio_ny_hoy * np.exp(retornos_sim.sum(axis=0))

precios_carga_simulados = [
    calcular_componentes_carga(p_ny, trm_hoy, LIBRAS_POR_CARGA, KG_PASILLA_POR_CARGA, PRECIO_PASILLA_COP_KG)[0]
    for p_ny in precios_ny_simulados
]

var_95 = np.percentile(precios_carga_simulados, 5)
promedio_sim = np.mean(precios_carga_simulados)

col_a, col_b = st.columns(2)
col_a.metric("Precio Promedio Esperado por Carga", f"${promedio_sim:,.0f} COP")
col_b.metric("Límite de Riesgo (VaR 95%)", f"${var_95:,.0f} COP", delta=f"-${precio_carga_hoy - var_95:,.0f}", delta_color="inverse")

# Gráfico
counts, bin_edges = np.histogram(precios_carga_simulados, bins=30)
chart_df = pd.DataFrame({"Escenarios": counts}, index=[f"${b:,.0f}" for b in bin_edges[:-1]])
st.bar_chart(chart_df)
