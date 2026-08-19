import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta

# Importamos las funciones de tu motor existente
from motor import obtener_datos_mercado, calcular_precio_interno_referencia, EWMA_LAMBDA


@st.cache_data(ttl=3600)  # Re-calcula cada hora o cuando cambien los parámetros
def ejecutar_simulacion_montecarlo(horizontes, simulaciones, factor_rendimiento):
    """
    Simula trayectorias diarias completas con Movimiento Browniano Geométrico
    y covarianza EWMA. Devuelve tanto los datos estadísticos como el objeto de la gráfica.
    """
    # 1. Obtener datos históricos
    df_mercado, _ = obtener_datos_mercado()
    
    # 2. Calcular retornos logarítmicos y parámetros EWMA
    retornos = np.log(df_mercado / df_mercado.shift(1)).dropna()
    
    media_diaria_cafe = retornos['cafe'].mean()
    media_diaria_trm = retornos['trm'].mean()
    
    alpha = 1 - EWMA_LAMBDA
    cov_ewma_serie = retornos.ewm(alpha=alpha, adjust=False).cov()
    ultima_fecha = retornos.index[-1]
    cov_actual = cov_ewma_serie.loc[ultima_fecha].values
    
    # Descomposición de Cholesky
    L = np.linalg.cholesky(cov_actual)
    
    # 3. Configuración inicial
    max_dias = max(horizontes)
    ultimo_cafe = df_mercado['cafe'].iloc[-1]
    ultima_trm = df_mercado['trm'].iloc[-1]
    
    trayectorias_cafe = np.zeros((max_dias + 1, simulaciones))
    trayectorias_trm = np.zeros((max_dias + 1, simulaciones))
    
    trayectorias_cafe[0, :] = ultimo_cafe
    trayectorias_trm[0, :] = ultima_trm
    
    # 4. Simulación Vectorizada/Iterativa
    for t in range(1, max_dias + 1):
        Z = np.random.standard_normal(size=(simulaciones, 2))
        choques = Z @ L.T
        
        drift_cafe = media_diaria_cafe - 0.5 * cov_actual[0, 0]
        trayectorias_cafe[t, :] = trayectorias_cafe[t-1, :] * np.exp(drift_cafe + choques[:, 0])
        
        drift_trm = media_diaria_trm - 0.5 * cov_actual[1, 1]
        trayectorias_trm[t, :] = trayectorias_trm[t-1, :] * np.exp(drift_trm + choques[:, 1])

    # 5. Transformar a Precio Interno (Carga)
    trayectorias_carga = calcular_precio_interno_referencia(
        trayectorias_cafe, 
        trayectorias_trm, 
        factor_rendimiento=factor_rendimiento
    )
    
    # 6. Métrica y Tabla
    precio_hoy = trayectorias_carga[0, 0]
    tabla_resultados = []
    
    for h in horizontes:
        precios_h = trayectorias_carga[h, :]
        p5 = np.percentile(precios_h, 5)
        p50 = np.percentile(precios_h, 50)
        p95 = np.percentile(precios_h, 95)
        
        tabla_resultados.append({
            "Horizonte": f"{h} Días",
            "Escenario Pesimista (P5%)": p5,
            "Predicción Esperada (P50%)": p50,
            "Escenario Optimista (P95%)": p95,
            "Variación Esperada (%)": ((p50 - precio_hoy) / precio_hoy) * 100
        })
        
    df_resumen = pd.DataFrame(tabla_resultados)
    
    # 7. Generación de la Gráfica Fan Chart
    dias_eje = np.arange(0, max_dias + 1)
    
    p5_diario = np.percentile(trayectorias_carga, 5, axis=1)
    p25_diario = np.percentile(trayectorias_carga, 25, axis=1)
    p50_diario = np.percentile(trayectorias_carga, 50, axis=1)
    p75_diario = np.percentile(trayectorias_carga, 75, axis=1)
    p95_diario = np.percentile(trayectorias_carga, 95, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    ax.fill_between(dias_eje, p5_diario, p95_diario, color='#1E3A8A', alpha=0.15, label='Incertidumbre (90% Confianza)')
    ax.fill_between(dias_eje, p25_diario, p75_diario, color='#1E3A8A', alpha=0.30, label='Zona de Alta Probabilidad (50% Confianza)')
    
    ax.plot(dias_eje, p50_diario, color='#2563EB', linewidth=2, label='Mediana Esperada')
    ax.axhline(precio_hoy, color='#10B981', linestyle='--', alpha=0.8, label=f'Spot Hoy (${precio_hoy:,.0f} COP)')
    
    for h in horizontes:
        ax.axvline(h, color='gray', linestyle=':', alpha=0.5)
        ax.plot(h, p50_diario[h], 'ro')
        ax.annotate(f"{h}d\n${p50_diario[h]:,.0f}", (h, p50_diario[h]),
                    textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8, weight='bold')

    ax.set_title(f"Abanico de Proyección Predictiva - Precio Interno (FR {int(factor_rendimiento)})", fontsize=11, weight='bold')
    ax.set_xlabel("Días Proyectados hacia el Futuro")
    ax.set_ylabel("Precio Carga (COP)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    
    return precio_hoy, df_resumen, fig


def renderizar_modulo_prediccion():
    st.header("📊 Análisis Predictivo Multitemporal (Precio Interno & TRM)")
    st.caption("Simulación Monte Carlo (10.000 trayectorias) basada en Movimiento Browniano Geométrico con volatilidad EWMA.")

    # Panel lateral o de control de la simulación
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    
    with col_ctrl1:
        factor_rend = st.number_input("Factor de Rendimiento (FR):", min_value=85.0, max_value=105.0, value=94.0, step=0.5)
    with col_ctrl2:
        num_sims = st.selectbox("Número de Simulaciones:", options=[1000, 5000, 10000, 20000], index=2)
    with col_ctrl3:
        st.write("") # Espaciador
        btn_ejecutar = st.button("🔄 Recalcular Proyección", use_container_width=True)

    if btn_ejecutar:
        st.cache_data.clear()

    # Ejecutar simulación
    with st.spinner("Ejecutando simulaciones estocásticas..."):
        precio_hoy, df_resumen, fig_fan = ejecutar_simulacion_montecarlo(
            horizontes=[30, 60, 90],
            simulaciones=num_sims,
            factor_rendimiento=factor_rend
        )

    # 1. Mostrar Precio Base Actual
    st.subheader(f"Precio Base Actual: **${precio_hoy:,.0f} COP** / Carga")
    st.divider()

    # 2. Métricas destacadas para 30, 60 y 90 días
    cols = st.columns(3)
    for idx, row in df_resumen.iterrows():
        with cols[idx]:
            variacion = row["Variación Esperada (%)"]
            st.metric(
                label=f"Proyección a {row['Horizonte']}",
                value=f"${row['Predicción Esperada (P50%)']:,.0f} COP",
                delta=f"{variacion:+.2f}%",
                delta_color="normal"
            )
            st.caption(f"📉 Min (P5%): **${row['Escenario Pesimista (P5%)']:,.0f}**")
            st.caption(f"📈 Max (P95%): **${row['Escenario Optimista (P95%)']:,.0f}**")

    # 3. Gráfica en Streamlit
    st.subheader("Abanico de Incertidumbre y Tendencia")
    st.pyplot(fig_fan)

    # 4. Tabla Detallada
    st.subheader("Resumen de Escenarios Predictivos")
    
    # Formatear la tabla para la vista
    df_format = df_resumen.copy()
    for col in ["Escenario Pesimista (P5%)", "Predicción Esperada (P50%)", "Escenario Optimista (P95%)"]:
        df_format[col] = df_format[col].map("${:,.2f} COP".format)
    df_format["Variación Esperada (%)"] = df_format["Variación Esperada (%)"].map("{:+.2f}%".format)
    
    st.dataframe(df_format, use_container_width=True, hide_index=True)


# Para correr como pestaña o componente principal:
if __name__ == "__main__":
    st.set_page_config(page_title="Predicción de Precios de Café", layout="wide")
    renderizar_modulo_prediccion()
