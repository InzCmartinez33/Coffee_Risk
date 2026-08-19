import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import os
from datetime import datetime, timedelta

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
st.markdown("Herramienta de simulación de Montecarlo y evaluación de coberturas financieras (Hedging) para exportadores de café en Colombia.")

# =====================================================================
# SIDEBAR / CONFIGURACIÓN DE PARÁMETROS INTERACTIVOS
# =====================================================================
st.sidebar.header("⚙️ Parámetros de Operación")

DIFERENCIAL_CALIDAD_USD_LB = st.sidebar.number_input(
    "Diferencial de Calidad (USD/lb)",
    value=0.08, step=0.01, format="%.2f",
    help="Prima o descuento sobre el contrato KC=F de la Bolsa de NY."
)

COSTOS_EXPORTACION_USD_LB = st.sidebar.number_input(
    "Costos de Exportación (USD/lb)",
    value=0.12, step=0.01, format="%.2f",
    help="Logística, trilla, empaque, comisiones, seguros, etc."
)

LIBRAS_POR_CARGA = st.sidebar.number_input(
    "Libras por Carga (125 kg)",
    value=96.25, step=0.25, format="%.2f",
    help="Libras de café verde excelso por carga de 125 kg."
)

VOLUMEN_CARGAS = st.sidebar.number_input(
    "Volumen de la Cosecha/Inventario (Cargas)",
    value=180, step=10, min_value=1
)

TENOR_ANALISIS = st.sidebar.selectbox(
    "Plazo / Horizonte de Análisis",
    options=["1m", "3m", "6m"],
    index=0,
    format_func=lambda x: {"1m": "1 Mes (30 días)", "3m": "3 Meses (90 días)", "6m": "6 Meses (180 días)"}[x]
)

TENOR_A_DIAS = {"1m": 30, "3m": 90, "6m": 180}
dias_analisis = TENOR_A_DIAS[TENOR_ANALISIS]

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Parámetros de Cobertura")

FORWARD_SPREAD_COP_USD = st.sidebar.number_input(
    "Spread Forward Bancario (COP/USD)",
    value=15.0, step=1.0, format="%.1f"
)

OPCION_PUT_PCT_PRIMA = st.sidebar.number_input(
    "Prima Teórica Put (% del portafolio)",
    value=4.0, step=0.5, format="%.1f"
) / 100.0

OPCION_PUT_COMISION_BROKER_USD = st.sidebar.number_input(
    "Comisión Broker Put (USD)",
    value=50.0, step=5.0
)

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

PERIODOS_A_DIAS = {
    "5d": 5, "1mo": 31, "3mo": 92, "6mo": 183,
    "1y": 366, "2y": 731, "5y": 1827, "10y": 3653,
    "ytd": 366, "max": 5000,
}

# =====================================================================
# FUNCIONES DE OBTENCIÓN Y PROCESAMIENTO DE DATOS
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

    if not registros:
        raise ValueError("La API oficial de TRM (datos.gov.co) no devolvió registros.")

    df = pd.DataFrame(registros)
    df["fecha"] = pd.to_datetime(df["vigenciadesde"]).dt.normalize()
    df["trm"] = df["valor"].astype(float)
    return df.set_index("fecha")[["trm"]]

def _normalizar_indice_fechas(indice):
    idx = pd.to_datetime(indice)
    if idx.tz is not None:
        idx = idx.tz_convert(None)
    return idx.normalize()

@st.cache_data(ttl=3600)
def obtener_datos_mercado(periodo="1y", cache_path="cache_mercado.csv"):
    dias_historia = PERIODOS_A_DIAS.get(periodo, 366)
    fecha_inicio = datetime.now() - timedelta(days=dias_historia)

    try:
        from curl_cffi import requests as curl_requests
        sesion_yahoo = curl_requests.Session(impersonate="chrome")
    except ImportError:
        sesion_yahoo = requests.Session()
        sesion_yahoo.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    try:
        t = yf.Ticker("KC=F", session=sesion_yahoo)
        df_cafe = t.history(period=periodo)

        if df_cafe.empty:
            raise ValueError("No se encontraron datos para el café (KC=F).")

        df_trm = obtener_trm_oficial(fecha_inicio)

        datos_cafe = df_cafe[["Close"]].rename(columns={"Close": "cafe"})
        datos_cafe.index = _normalizar_indice_fechas(datos_cafe.index)

        datos = datos_cafe.join(df_trm, how="inner").dropna()
        if datos.empty:
            raise ValueError("No hubo fechas en común entre el café y la TRM.")

        try:
            datos.to_csv(cache_path)
        except Exception:
            pass

        return datos, {"fuente": "yahoo_finance + datos.gov.co", "timestamp": datetime.now()}

    except Exception as e:
        if os.path.exists(cache_path):
            timestamp_cache = datetime.fromtimestamp(os.path.getmtime(cache_path))
            datos_cache = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return datos_cache, {"fuente": "cache_local", "timestamp": timestamp_cache}
        raise RuntimeError(f"Error descargando datos y no hay caché local: {e}")

def calcular_precio_interno_referencia(precio_ny_centavos, trm):
    precio_ny_usd_lb = precio_ny_centavos / 100
    precio_neto_usd_lb = precio_ny_usd_lb + DIFERENCIAL_CALIDAD_USD_LB - COSTOS_EXPORTACION_USD_LB
    precio_carga_cop = precio_neto_usd_lb * trm * LIBRAS_POR_CARGA
    return precio_carga_cop

def simular_montecarlo_carga(datos_historicos, dias_proyeccion=90, simulaciones=5000, semilla=42):
    if semilla is not None:
        np.random.seed(semilla)

    retornos = np.log(datos_historicos / datos_historicos.shift(1)).dropna()
    media = retornos.mean().values
    covarianza = retornos.cov().values

    shocks = np.random.multivariate_normal(media, covarianza, size=(dias_proyeccion, simulaciones))
    log_retornos_acumulados = shocks.sum(axis=0)

    ultimo_cafe = datos_historicos['cafe'].iloc[-1]
    ultima_trm = datos_historicos['trm'].iloc[-1]

    cafe_final = ultimo_cafe * np.exp(log_retornos_acumulados[:, 0])
    trm_final = ultima_trm * np.exp(log_retornos_acumulados[:, 1])

    return calcular_precio_interno_referencia(cafe_final, trm_final)

def simular_forward(precios_carga_simulados, precio_carga_actual, trm_actual):
    dolares_por_carga = precio_carga_actual / trm_actual
    costo_transaccion_por_carga = FORWARD_SPREAD_COP_USD * dolares_por_carga
    precio_pactado = precio_carga_actual - costo_transaccion_por_carga
    escenarios_netos = np.full_like(precios_carga_simulados, precio_pactado, dtype=float)
    return escenarios_netos, precio_pactado, costo_transaccion_por_carga

def simular_opcion_put(precios_carga_simulados, precio_carga_actual, trm_actual):
    strike_price = precio_carga_actual
    valor_total_portafolio = precio_carga_actual * VOLUMEN_CARGAS
    costo_prima_total_cop = valor_total_portafolio * OPCION_PUT_PCT_PRIMA
    comision_total_cop = OPCION_PUT_COMISION_BROKER_USD * trm_actual
    costo_seguro_por_carga = (costo_prima_total_cop + comision_total_cop) / VOLUMEN_CARGAS

    precios_carga_simulados = np.asarray(precios_carga_simulados, dtype=float)
    payoff = np.maximum(strike_price - precios_carga_simulados, 0.0)
    escenarios_netos = precios_carga_simulados + payoff - costo_seguro_por_carga

    return escenarios_netos, strike_price, costo_seguro_por_carga

def simular_opcion_put_mercado(precios_carga_simulados, strike, prima):
    precios_carga_simulados = np.asarray(precios_carga_simulados, dtype=float)
    payoff = np.maximum(strike - precios_carga_simulados, 0.0)
    return precios_carga_simulados + payoff - prima

def evaluar_opciones_put_mercado(precios_carga_simulados, tenor):
    opciones = COTIZACIONES_PUT_COP.get(tenor, [])
    resultados = []
    for opcion in opciones:
        strike = opcion["strike"]
        prima = opcion["prima"]
        escenarios = simular_opcion_put_mercado(precios_carga_simulados, strike, prima)
        resultados.append({
            "strike": strike,
            "prima": prima,
            "piso_neto": strike - prima,
            "promedio": np.mean(escenarios),
            "var_95": np.percentile(escenarios, 5),
            "peor_caso": np.min(escenarios),
            "mejor_caso": np.max(escenarios),
        })
    return resultados

# =====================================================================
# EJECUCIÓN PRINCIPAL EN STREAMLIT
# =====================================================================
with st.spinner("Conectando con fuentes financieras y procesando mercado..."):
    try:
        df_mercado, info_datos = obtener_datos_mercado()
        exito = True
    except Exception as e:
        st.error(f"Error al cargar datos de mercado: {e}")
        exito = False

if exito:
    precio_ny_hoy = df_mercado['cafe'].iloc[-1]
    trm_hoy = df_mercado['trm'].iloc[-1]

    precio_carga_hoy = calcular_precio_interno_referencia(precio_ny_hoy, trm_hoy)
    valor_inventario_hoy = precio_carga_hoy * VOLUMEN_CARGAS

    if info_datos["fuente"] == "cache_local":
        st.warning(f"⚠️ Usando caché local ({info_datos['timestamp'].strftime('%Y-%m-%d %H:%M')}).")
    else:
        st.success("✅ Datos de mercado actualizados en tiempo real.")

    # 1. Métricas Principales de Mercado
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Café NY (KC=F)", f"{precio_ny_hoy/100:.2f} USD/lb")
    col2.metric("TRM Oficial (COP)", f"${trm_hoy:,.2f}")
    col3.metric("Precio Carga Ref.", f"${precio_carga_hoy:,.0f} COP")
    col4.metric("Portafolio Total", f"${valor_inventario_hoy/1e6:,.2f} M COP")

    st.markdown("---")

    # 2. Montecarlo y VaR
    st.subheader(f"📊 Análisis de Riesgo Financiero (Montecarlo {dias_analisis} días)")

    precios_futuros = simular_montecarlo_carga(df_mercado, dias_proyeccion=dias_analisis)

    var_95 = np.percentile(precios_futuros, 5)
    precio_promedio_sim = np.mean(precios_futuros)
    perdida_max_por_carga = precio_carga_hoy - var_95
    perdida_total_empresa = perdida_max_por_carga * VOLUMEN_CARGAS

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Precio Promedio Simulado", f"${precio_promedio_sim:,.0f} COP")
    col_b.metric("Límite Crítico (VaR 95%)", f"${var_95:,.0f} COP", delta=f"-${perdida_max_por_carga:,.0f}", delta_color="inverse")
    col_c.metric("Riesgo Máximo Empresa", f"${perdida_total_empresa/1e6:,.2f} M COP", delta_color="inverse")

    if perdida_total_empresa > (valor_inventario_hoy * 0.07):
        st.error("🔴 **RECOMENDACIÓN:** RIESGO ALTO. Se aconseja ejecutar estrategias de cobertura (Hedging).")
    else:
        st.success("🟢 **RECOMENDACIÓN:** RIESGO TOLERABLE. Monitorear volatilidad de mercado.")

    st.markdown("---")

    # Gráfico de Distribución de Escenarios
    st.subheader("📈 Distribución de Escenarios Simulados")
    counts, bin_edges = np.histogram(precios_futuros, bins=40)
    chart_df = pd.DataFrame({"Frecuencia": counts}, index=[f"${b:,.0f}" for b in bin_edges[:-1]])
    st.bar_chart(chart_df)

    st.markdown("---")

    # 3. Coberturas Teóricas
    st.subheader("🛡️ Comparación de Coberturas Teóricas")

    escenarios_forward, precio_pactado, costo_forward = simular_forward(precios_futuros, precio_carga_hoy, trm_hoy)
    escenarios_put, strike_put, costo_put = simular_opcion_put(precios_futuros, precio_carga_hoy, trm_hoy)

    tab1, tab2, tab3 = st.tabs(["Sin Cobertura (Spot)", "Forward NDF", "Opción Put (Teórica)"])

    with tab1:
        st.write("**Posición expuesta 100% al mercado**")
        st.write(f"- **Ingreso Promedio:** ${np.mean(precios_futuros):,.0f} COP/carga")
        st.write(f"- **Piso (VaR 95%):** ${var_95:,.0f} COP/carga")
        st.write(f"- **Rango:** ${np.min(precios_futuros):,.0f} - ${np.max(precios_futuros):,.0f} COP/carga")

    with tab2:
        st.write("**Fijación de precio total**")
        st.write(f"- **Precio Pactado Garantizado:** ${precio_pactado:,.0f} COP/carga")
        st.write(f"- **Costo Spread Bancario:** ${costo_forward:,.0f} COP/carga")
        st.write(f"- **Ingreso Total Garantizado:** ${(precio_pactado * VOLUMEN_CARGAS)/1e6:,.2f} M COP")

    with tab3:
        st.write("**Seguro de precio mínimo**")
        st.write(f"- **Piso Neto Garantizado:** ${(strike_put - costo_put):,.0f} COP/carga")
        st.write(f"- **Costo Prima Estimado:** ${costo_put:,.0f} COP/carga")
        st.write(f"- **Ingreso Promedio Neto:** ${np.mean(escenarios_put):,.0f} COP/carga")

    st.markdown("---")

    # 4. Cotizaciones Reales
    st.subheader(f"💰 Cotizaciones Reales de Mercado para Opciones Put ({TENOR_ANALISIS.upper()})")

    opciones_reales = evaluar_opciones_put_mercado(precios_futuros, tenor=TENOR_ANALISIS)
    df_opciones = pd.DataFrame(opciones_reales)

    if not df_opciones.empty:
        df_opciones_display = pd.DataFrame({
            "Strike (COP)": df_opciones["strike"].apply(lambda x: f"${x:,.0f}"),
            "Prima (COP)": df_opciones["prima"].apply(lambda x: f"${x:,.0f}"),
            "Piso Neto (COP)": df_opciones["piso_neto"].apply(lambda x: f"${x:,.0f}"),
            "Promedio Simulado": df_opciones["promedio"].apply(lambda x: f"${x:,.0f}"),
            "VaR 95% Cobertura": df_opciones["var_95"].apply(lambda x: f"${x:,.0f}"),
            "Costo Seguro Total": (df_opciones["prima"] * VOLUMEN_CARGAS).apply(lambda x: f"${x/1e6:,.2f} M COP")
        })
        st.dataframe(df_opciones_display, use_container_width=True)
    else:
        st.info("No hay cotizaciones disponibles para este tenor.")
