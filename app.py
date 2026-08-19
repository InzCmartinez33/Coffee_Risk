import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importar funciones de los módulos locales
from datos import cargar_datos_mercado
from proyecciones import calcular_todas_las_proyecciones

# ---------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Gestión de Riesgo Financiero - Café",
    page_icon="☕",
    layout="wide"
)

st.title("☕ Modelo de Gestión de Riesgo y Cobertura Cafetera")
st.markdown("""
Esta herramienta simula escenarios de precio para la **carga de café en Colombia** considerando 
la volatilidad estocástica conjunta (GBM-EWMA) de la **Bolsa de Nueva York (C C) ** y la **TRM (COP/USD)**.
""")

# ---------------------------------------------------------------------
# BARRA LATERAL - PARÁMETROS DE ENTRADA
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Parámetros de Mercado y Operación")

# Carga de datos de mercado
df_mercado = cargar_datos_mercado()

if df_mercado.empty:
    st.error("Error al cargar los datos de mercado. Verifica la fuente de datos.")
    st.stop()

# Últimos valores de mercado observados
precio_ny_hoy = df_mercado['cafe'].iloc[-1]  # Centavos USD / lb
trm_hoy = df_mercado['trm'].iloc[-1]        # COP / USD

st.sidebar.subheader("Valores Actuales de Mercado")
st.sidebar.info(f"**Café NY:** {precio_ny_hoy:.2f} ¢/lb\n\n**TRM:** ${trm_hoy:,.2f} COP")

# Inputs modificables por el usuario
st.sidebar.subheader("Variables Operativas")
diferencial_usd = st.sidebar.number_input("Diferencial (USD/lb)", value=0.15, step=0.01)
costos_usd = st.sidebar.number_input("Costos de Exportación (USD/lb)", value=0.08, step=0.01)
libras_carga = st.sidebar.number_input("Libras por Carga", value=275.58, step=0.01)
kg_pasilla = st.sidebar.number_input("Kg Pasilla por Carga", value=6.0, step=0.5)
precio_pasilla_kg = st.sidebar.number_input("Precio Pasilla (COP/kg)", value=7000, step=500)

st.sidebar.subheader("Escenario de Análisis")
tenores_disponibles = ["1 Mes", "2 Meses", "3 Meses"]
tenor_seleccionado = st.sidebar.selectbox("Plazo de Cobertura", tenores_disponibles)

mapa_dias = {"1 Mes": 30, "2 Meses": 60, "3 Meses": 90}
dias_analisis = mapa_dias[tenor_seleccionado]

VOLUMEN_CARGAS = st.sidebar.number_input("Volumen a Proteger (Cargas)", value=1000, step=100)

# Cotizaciones de Opciones Put por Tenor
cotizaciones_opciones = {
    "1 Mes": [
        {"strike": 2100000, "prima": 45000},
        {"strike": 2200000, "prima": 75000},
        {"strike": 2300000, "prima": 115000}
    ],
    "2 Meses": [
        {"strike": 2100000, "prima": 65000},
        {"strike": 2200000, "prima": 105000},
        {"strike": 2300000, "prima": 155000}
    ],
    "3 Meses": [
        {"strike": 2100000, "prima": 85000},
        {"strike": 2200000, "prima": 130000},
        {"strike": 2300000, "prima": 185000}
    ]
}

params = {
    "diferencial_usd": diferencial_usd,
    "costos_usd": costos_usd,
    "libras_carga": libras_carga,
    "kg_pasilla": kg_pasilla,
    "precio_pasilla_kg": precio_pasilla_kg,
    "dias_analisis": dias_analisis,
    "volumen_cargas": VOLUMEN_CARGAS,
    "tenor": tenor_seleccionado,
    "cotizaciones": cotizaciones_opciones
}

# ---------------------------------------------------------------------
# 1. EJECUCIÓN DEL MODELO DE PROYECCIONES
# ---------------------------------------------------------------------
dict_proy = calcular_todas_las_proyecciones(params, df_mercado)

# Extracción de variables desde el diccionario retornado por proyecciones.py
trayectorias_carga = dict_proy["trayectorias_carga"]
trayectorias_cafe = dict_proy["trayectorias_cafe"]
trayectorias_trm = dict_proy["trayectorias_trm"]
precios_futuros = dict_proy["precios_futuros"]
var_95 = dict_proy["var_95"]
precio_promedio_sim = dict_proy["precio_promedio_sim"]
trm_promedio_sim = dict_proy["trm_promedio_sim"]
df_cotizaciones = dict_proy["df_cotizaciones"]

# Trayectorias en el día final simulado
cafe_futuro = trayectorias_cafe[-1, :]
trm_futura = trayectorias_trm[-1, :]

# Precio base de la carga hoy
precio_carga_hoy = (
    ((precio_ny_hoy / 100.0) + diferencial_usd - costos_usd) * trm_hoy * libras_carga
) + (kg_pasilla * precio_pasilla_kg)

valor_inventario_hoy = precio_carga_hoy * VOLUMEN_CARGAS

# ---------------------------------------------------------------------
# METRICAS DE CABECERA (ESTADO ACTUAL)
# ---------------------------------------------------------------------
st.subheader("1. Estado Actual de Valoración")
c1, c2, c3 = st.columns(3)
c1.metric("Precio Carga Referencia Hoy", f"${precio_carga_hoy:,.0f} COP")
c2.metric("Volumen Inventario/Cosecha", f"{VOLUMEN_CARGAS:,.0f} Cargas")
c3.metric("Valor Total del Inventario", f"${valor_inventario_hoy/1e6:,.2f} M COP")

st.markdown("---")

# ---------------------------------------------------------------------
# 2. PROYECCIONES INDIVIDUALES (CAFÉ Y TRM)
# ---------------------------------------------------------------------
st.subheader(f"2. Proyecciones Individuales de Mercado a {dias_analisis} Días")

if len(cafe_futuro) > 0 and len(trm_futura) > 0:
    # Usar directamente el promedio retornado por proyecciones.py para mantener consistencia
    trm_prom_sim = trm_promedio_sim
    cafe_prom_sim = np.mean(cafe_futuro) / 100.0  # Promedio en USD/lb

    # Escenarios extremos
    cafe_p5 = np.percentile(cafe_futuro, 5) / 100.0
    cafe_p95 = np.percentile(cafe_futuro, 95) / 100.0

    trm_p5 = np.percentile(trm_futura, 5)
    trm_p95 = np.percentile(trm_futura, 95)

    col_cafe, col_trm = st.columns(2)

    with col_cafe:
        st.markdown("#### ☕ Proyección Café NY (USD/lb)")
        st.metric(
            "Precio Promedio Simulado", 
            f"${cafe_prom_sim:.2f} USD/lb", 
            delta=f"{(cafe_prom_sim - (precio_ny_hoy/100)):.2f} USD"
        )
        st.caption(f"📉 **Escenario Crítico (P5%):** ${cafe_p5:.2f} USD/lb")
        st.caption(f"📈 **Escenario Alcista (P95%):** ${cafe_p95:.2f} USD/lb")

        fig_c, ax_c = plt.subplots(figsize=(6, 3))
        ax_c.hist(cafe_futuro / 100.0, bins=40, color='#B45309', alpha=0.7, edgecolor='white')
        ax_c.axvline(precio_ny_hoy/100, color='#10B981', linestyle='--', label='Hoy')
        ax_c.axvline(cafe_prom_sim, color='#1E3A8A', linestyle='-', label='Promedio Proyectado')
        ax_c.set_title("Distribución Café NY", fontsize=9, fontweight='bold')
        ax_c.set_xlabel("USD / lb", fontsize=8)
        ax_c.grid(True, alpha=0.3)
        ax_c.legend(fontsize=7)
        st.pyplot(fig_c)

    with col_trm:
        st.markdown("#### 💵 Proyección Dólar TRM (COP/USD)")
        st.metric(
            "TRM Promedio Simulada", 
            f"${trm_prom_sim:,.2f} COP", 
            delta=f"{(trm_prom_sim - trm_hoy):,.2f} COP"
        )
        st.caption(f"📉 **Escenario Crítico (P5%):** ${trm_p5:,.2f} COP")
        st.caption(f"📈 **Escenario Alcista (P95%):** ${trm_p95:,.2f} COP")

        fig_t, ax_t = plt.subplots(figsize=(6, 3))
        ax_t.hist(trm_futura, bins=40, color='#047857', alpha=0.7, edgecolor='white')
        ax_t.axvline(trm_hoy, color='#10B981', linestyle='--', label='Hoy')
        ax_t.axvline(trm_prom_sim, color='#1E3A8A', linestyle='-', label='Promedio Proyectado')
        ax_t.set_title("Distribución TRM", fontsize=9, fontweight='bold')
        ax_t.set_xlabel("COP / USD", fontsize=8)
        ax_t.grid(True, alpha=0.3)
        ax_t.legend(fontsize=7)
        st.pyplot(fig_t)

st.markdown("---")

# ---------------------------------------------------------------------
# 3. MONTECARLO Y VAR PRECIO DE LA CARGA
# ---------------------------------------------------------------------
st.subheader(f"3. Análisis de Riesgo Financiero - Carga de Café a {dias_analisis} Días")

perdida_max_por_carga = precio_carga_hoy - var_95
perdida_total_empresa = perdida_max_por_carga * VOLUMEN_CARGAS

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Precio Carga Promedio", f"${precio_promedio_sim:,.0f} COP")
col_b.metric("TRM Promedio Simulada", f"${trm_promedio_sim:,.2f} COP")
col_c.metric("Límite Crítico (VaR 95%)", f"${var_95:,.0f} COP", delta=f"-${perdida_max_por_carga:,.0f}", delta_color="inverse")
col_d.metric("Riesgo Máximo Empresa", f"${perdida_total_empresa/1e6:,.2f} M COP", delta_color="inverse")

if perdida_total_empresa > (valor_inventario_hoy * 0.07):
    st.error("🔴 **RECOMENDACIÓN:** RIESGO ALTO. Se aconseja tomar coberturas financieras (Forward u Opciones Put).")
else:
    st.success("🟢 **RECOMENDACIÓN:** RIESGO TOLERABLE. Monitorear volatilidad de mercado.")

# Gráfico del Histograma de Carga
fig, ax = plt.subplots(figsize=(10, 4))
n, bins, patches = ax.hist(precios_futuros, bins=60, density=True, alpha=0.6, color='#2563EB', edgecolor='white')

for i in range(len(patches)):
    if bins[i] < var_95:
        patches[i].set_facecolor('#DC2626')

ax.axvline(precio_carga_hoy, color='#10B981', linestyle='--', linewidth=2, label=f'Precio Base Hoy (${precio_carga_hoy:,.0f})')
ax.axvline(var_95, color='#DC2626', linestyle='-', linewidth=2, label=f'VaR 95% (${var_95:,.0f})')
ax.axvline(precio_promedio_sim, color='#1E3A8A', linestyle=':', linewidth=2, label=f'Promedio Simulado (${precio_promedio_sim:,.0f})')

ax.set_title(f"Distribución Carga Total a {dias_analisis} Días", fontsize=11, fontweight='bold')
ax.set_xlabel("Precio de la Carga (COP)")
ax.set_ylabel("Densidad de Probabilidad")
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right')
ax.xaxis.set_major_formatter('${x:,.0f}')

st.pyplot(fig)

st.markdown("---")

# ---------------------------------------------------------------------
# 4. ESTRATEGIAS DE COBERTURA CON OPCIONES (PUTS)
# ---------------------------------------------------------------------
st.subheader(f"4. Evaluación de Cobertura con Opciones Put ({tenor_seleccionado})")

st.markdown("""
A continuación se evalúan las alternativas de seguro de precio (**Opciones Put**) disponibles para el plazo seleccionado. 
El **Piso Neto** representa el ingreso mínimo garantizado por carga tras descontar el costo de la prima.
""")

st.dataframe(df_cotizaciones, use_container_width=True)
