import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Importación del backend con la metodología GBM + EWMA
from proyecciones import calcular_precio_interno_referencia, calcular_todas_las_proyecciones

# =====================================================================
# CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT
# =====================================================================
st.set_page_config(
    page_title="RiskCafe Engine - Análisis de Riesgo Financiero",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("☕ RiskCafe Engine - Gestión de Riesgo para Café")
st.markdown("Simulación de Montecarlo con **Movimiento Browniano Geométrico (GBM)** y **Volatilidad Estructural EWMA**.")

# =====================================================================
# SIDEBAR / CONFIGURACIÓN DE PARÁMETROS INTERACTIVOS
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
    LIBRAS_POR_CARGA = (125.0 / FACTOR_RENDIMIENTO) * 70.0 * 2.20462262
    KG_EXCELSO_POR_CARGA = (125.0 / FACTOR_RENDIMIENTO) * 70.0
    KG_PASILLA_POR_CARGA = max(0.0, 125.0 - KG_EXCELSO_POR_CARGA - 18.0)
    
    st.sidebar.info(f"💡 **Excelso:** {LIBRAS_POR_CARGA:.2f} lbs ({KG_EXCELSO_POR_CARGA:.1f} kg) | **Pasilla:** {KG_PASILLA_POR_CARGA:.1f} kg")
else:
    LIBRAS_POR_CARGA = st.sidebar.number_input(
        "Libras de Excelso por Carga (125 kg)",
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

TENOR_ANALISIS = st.sidebar.selectbox("Plazo / Horizonte de Análisis", options=["1m", "3m", "6m"], index=0)
dias_analisis = {"1m": 30, "3m": 90, "6m": 180}[TENOR_ANALISIS]

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Parámetros de Cobertura")
FORWARD_SPREAD_COP_USD = st.sidebar.number_input("Spread Forward (COP/USD)", value=15.0, step=1.0)
OPCION_PUT_PCT_PRIMA = st.sidebar.number_input("Prima Teórica Put (% portafolio)", value=4.0, step=0.5) / 100.0
OPCION_PUT_COMISION_BROKER_USD = st.sidebar.number_input("Comisión Broker Put (USD)", value=50.0, step=5.0)

# =====================================================================
# COTIZACIONES REALES DE MERCADO — OPCIONES PUT
# =====================================================================
COTIZACIONES_PUT_COP = {
    "1m": [
        {"strike": 2_420_000, "prima": 168_000},
        {"strike": 2_520_000, "prima": 221_000},
        {"strike": 2_620_000, "prima": 221_000},
    ],
    "3m": [
        {"strike": 2_420_000, "prima": 168_000},
        {"strike": 2_520_000, "prima": 221_000},
        {"strike": 2_620_000, "prima": 221_000},
    ],
    "6m": [
        {"strike": 2_420_000, "prima": 168_000},
        {"strike": 2_520_000, "prima": 221_000},
        {"strike": 2_620_000, "prima": 221_000},
    ],
}

# =====================================================================
# FUNCIONES DE OBTENCIÓN DE DATOS
# =====================================================================
@st.cache_data(ttl=3600)
def obtener_trm_oficial(fecha_inicio):
    url = "https://www.datos.gov.co/resource/32sa-8pi3.json"
    params = {
        "$where": f"vigenciadesde >= '{fecha_inicio.strftime('%Y-%m-%dT00:00:00.000')}'",
        "$order": "vigenciadesde ASC",
        "$limit": 2000,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    registros = resp.json()
    df = pd.DataFrame(registros)
    df["fecha"] = pd.to_datetime(df["vigenciadesde"]).dt.normalize()
    df["trm"] = df["valor"].astype(float)
    return df.set_index("fecha")[["trm"]]

@st.cache_data(ttl=3600)
def obtener_datos_mercado():
    fecha_inicio = datetime.now() - timedelta(days=366)
    try:
        t = yf.Ticker("KC=F")
        df_cafe = t.history(period="1y")
        df_trm = obtener_trm_oficial(fecha_inicio)

        datos_cafe = df_cafe[["Close"]].rename(columns={"Close": "cafe"})
        datos_cafe.index = pd.to_datetime(datos_cafe.index).tz_localize(None).normalize()

        datos = datos_cafe.join(df_trm, how="inner").dropna()
        return datos, "OK"
    except Exception:
        fechas = pd.date_range(end=datetime.now(), periods=250, freq='B')
        np.random.seed(42)
        p_cafe = 245.5 + np.cumsum(np.random.normal(0, 2, 250))
        p_trm = 4050.0 + np.cumsum(np.random.normal(0, 15, 250))
        return pd.DataFrame({"cafe": p_cafe, "trm": p_trm}, index=fechas), "FALLBACK"

@st.cache_data(ttl=600, show_spinner=False)
def ejecutar_motor_proyecciones(params_dict, df_mercado):
    return calcular_todas_las_proyecciones(params_dict, df_mercado)

# =====================================================================
# EJECUCIÓN PRINCIPAL
# =====================================================================
with st.spinner("Conectando con mercados y procesando datos..."):
    df_mercado, estado = obtener_datos_mercado()

precio_ny_hoy = df_mercado['cafe'].iloc[-1]
trm_hoy = df_mercado['trm'].iloc[-1]

precio_carga_hoy, valor_excelso_hoy, valor_pasilla_hoy = calcular_precio_interno_referencia(
    precio_ny_hoy, trm_hoy, DIFERENCIAL_CALIDAD_USD_LB, COSTOS_EXPORTACION_USD_LB, 
    LIBRAS_POR_CARGA, KG_PASILLA_POR_CARGA, PRECIO_PASILLA_COP_KG
)
valor_inventario_hoy = precio_carga_hoy * VOLUMEN_CARGAS

# EJECUCIÓN DEL MOTOR INTERNO DE PROYECCIONES
params_simulacion = {
    'diferencial_usd': DIFERENCIAL_CALIDAD_USD_LB,
    'costos_usd': COSTOS_EXPORTACION_USD_LB,
    'libras_carga': LIBRAS_POR_CARGA,
    'kg_pasilla': KG_PASILLA_POR_CARGA,
    'precio_pasilla_kg': PRECIO_PASILLA_COP_KG,
    'dias_analisis': dias_analisis,
    'volumen_cargas': VOLUMEN_CARGAS,
    'tenor': TENOR_ANALISIS,
    'cotizaciones': COTIZACIONES_PUT_COP
}

with st.spinner("Ejecutando simulación GBM-EWMA en segundo plano..."):
    proyecciones = ejecutar_motor_proyecciones(params_simulacion, df_mercado)

precios_futuros = proyecciones['precios_futuros']
var_95 = proyecciones['var_95']
precio_promedio_sim = proyecciones['precio_promedio_sim']
trm_promedio_sim = proyecciones['trm_promedio_sim']
trayectorias_carga = proyecciones['trayectorias_carga']

# ---------------------------------------------------------------------
# 1. MÉTRICAS PRINCIPALES DE MERCADO
# ---------------------------------------------------------------------
st.subheader("1. Métricas Principales de Mercado")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Café NY (KC=F)", f"{precio_ny_hoy/100:.2f} USD/lb")
col2.metric("TRM Oficial (COP)", f"${trm_hoy:,.2f}")
col3.metric("Precio Carga Ref.", f"${precio_carga_hoy:,.0f} COP")
col4.metric("Portafolio Total", f"${valor_inventario_hoy/1e6:,.2f} M COP")

st.info(
    f"📌 **Desglose de la Carga de 125 kg:** "
    f"Valor Excelso: **${valor_excelso_hoy:,.0f} COP** | "
    f"Valor Pasilla Loca ({KG_PASILLA_POR_CARGA:.1f} kg @ ${PRECIO_PASILLA_COP_KG:,.0f}/kg): **${valor_pasilla_hoy:,.0f} COP**"
)

st.markdown("---")

# ---------------------------------------------------------------------
# 2. MONTECARLO Y VAR (METODOLOGÍA GBM - EWMA)
# ---------------------------------------------------------------------
st.subheader(f"2. Análisis de Riesgo Financiero (Montecarlo GBM-EWMA a {dias_analisis} días)")

perdida_max_por_carga = precio_carga_hoy - var_95
perdida_total_empresa = perdida_max_por_carga * VOLUMEN_CARGAS

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Precio Promedio Simulado", f"${precio_promedio_sim:,.0f} COP")
col_b.metric("TRM Promedio Simulada", f"${trm_promedio_sim:,.2f} COP")
col_c.metric("Límite Crítico (VaR 95%)", f"${var_95:,.0f} COP", delta=f"-${perdida_max_por_carga:,.0f}", delta_color="inverse")
col_d.metric("Riesgo Máximo Empresa", f"${perdida_total_empresa/1e6:,.2f} M COP", delta_color="inverse")

if perdida_total_empresa > (valor_inventario_hoy * 0.07):
    st.error("🔴 **RECOMENDACIÓN:** RIESGO ALTO. Se aconseja tomar coberturas financieras (Forward u Opciones Put).")
else:
    st.success("🟢 **RECOMENDACIÓN:** RIESGO TOLERABLE. Monitorear volatilidad de mercado.")

# ---------------------------------------------------------------------
# ABANICO DE PREDICCIÓN (FAN CHART)
# ---------------------------------------------------------------------
st.markdown("#### 📈 Abanico de Predicción Multitemporal (Fan Chart)")

dias_eje = np.arange(0, dias_analisis + 1)
p5_diario = np.percentile(trayectorias_carga, 5, axis=1)
p25_diario = np.percentile(trayectorias_carga, 25, axis=1)
p50_diario = np.percentile(trayectorias_carga, 50, axis=1)
p75_diario = np.percentile(trayectorias_carga, 75, axis=1)
p95_diario = np.percentile(trayectorias_carga, 95, axis=1)

fig = go.Figure()

# Banda 90% Confianza (P5 a P95)
fig.add_trace(go.Scatter(
    x=np.concatenate([dias_eje, dias_eje[::-1]]),
    y=np.concatenate([p95_diario, p5_diario[::-1]]),
    fill='toself',
    fillcolor='rgba(30, 58, 138, 0.15)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    name="Estrés / Volatilidad (90%)"
))

# Banda 50% Confianza (P25 a P75)
fig.add_trace(go.Scatter(
    x=np.concatenate([dias_eje, dias_eje[::-1]]),
    y=np.concatenate([p75_diario, p25_diario[::-1]]),
    fill='toself',
    fillcolor='rgba(30, 58, 138, 0.30)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    name="Alta Probabilidad (50%)"
))

# Mediana (P50)
fig.add_trace(go.Scatter(
    x=dias_eje, y=p50_diario,
    mode='lines',
    line=dict(color='#2563EB', width=3),
    name="Tendencia Esperada (P50%)"
))

# Línea base hoy
fig.add_shape(
    type="line", x0=0, x1=dias_analisis, y0=precio_carga_hoy, y1=precio_carga_hoy,
    line=dict(color="#10B981", width=2, dash="dash")
)

fig.update_layout(
    title=f"Proyección de Precios de Carga a {dias_analisis} Días",
    xaxis_title="Días Proyectados",
    yaxis_title="Precio Carga (COP)",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------
# 3. COBERTURAS TEÓRICAS
# ---------------------------------------------------------------------
st.subheader("3. Comparación de Coberturas Teóricas")
dolares_por_carga = precio_carga_hoy / trm_hoy
costo_forward = FORWARD_SPREAD_COP_USD * dolares_por_carga
precio_forward = precio_carga_hoy - costo_forward

costo_put = (precio_carga_hoy * VOLUMEN_CARGAS * OPCION_PUT_PCT_PRIMA + OPCION_PUT_COMISION_BROKER_USD * trm_hoy) / VOLUMEN_CARGAS
escenarios_put = np.maximum(precio_carga_hoy - precios_futuros, 0.0) + precios_futuros - costo_put

tab1, tab2, tab3 = st.tabs(["Sin Cobertura (Spot)", "Forward NDF", "Opción Put (Teórica)"])

with tab1:
    st.write("**Posición expuesta 100% a la volatilidad del mercado**")
    st.write(f"- **Piso (VaR 95%):** ${var_95:,.0f} COP/carga")
    st.write(f"- **Promedio Esperado:** ${precio_promedio_sim:,.0f} COP/carga")
    st.write(f"- **Rango:** ${np.min(precios_futuros):,.0f} - ${np.max(precios_futuros):,.0f} COP/carga")

with tab2:
    st.write("**Fijación total del precio (Elimina riesgo a la baja y alza)**")
    st.write(f"- **Precio Pactado Garantizado:** ${precio_forward:,.0f} COP/carga")
    st.write(f"- **Spread Bancario Aplicado:** ${costo_forward:,.0f} COP/carga")
    st.write(f"- **Valor Total Protegido:** ${(precio_forward * VOLUMEN_CARGAS)/1e6:,.2f} M COP")

with tab3:
    st.write("**Seguro de precio mínimo (Conserva ganancias si el café sube)**")
    st.write(f"- **Piso Neto Garantizado:** ${(precio_carga_hoy - costo_put):,.0f} COP/carga")
    st.write(f"- **Costo Prima Estimado:** ${costo_put:,.0f} COP/carga")
    st.write(f"- **Ingreso Promedio Neto:** ${np.mean(escenarios_put):,.0f} COP/carga")

st.markdown("---")

# ---------------------------------------------------------------------
# 4. COTIZACIONES REALES DE OPCIONES PUT
# ---------------------------------------------------------------------
st.subheader(f"4. Cotizaciones Reales de Mercado para Opciones Put ({TENOR_ANALISIS.upper()})")
st.dataframe(proyecciones['df_cotizaciones'], use_container_width=True)
