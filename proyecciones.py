import pandas as pd
import numpy as np

EWMA_LAMBDA = 0.94 

def calcular_precio_interno_referencia(
    precio_ny_centavos, trm, diferencial_usd, costos_usd, 
    libras_carga, kg_pasilla, precio_pasilla_kg
):
    precio_ny_usd_lb = precio_ny_centavos / 100.0
    precio_neto_usd_lb = precio_ny_usd_lb + diferencial_usd - costos_usd
    valor_excelso_cop = precio_neto_usd_lb * trm * libras_carga
    valor_pasilla_cop = kg_pasilla * precio_pasilla_kg
    precio_carga_total_cop = valor_excelso_cop + valor_pasilla_cop
    return precio_carga_total_cop, valor_excelso_cop, valor_pasilla_cop


def simular_gbm_ewma_trayectorias(
    df_mercado, 
    diferencial_usd, 
    costos_usd, 
    libras_carga, 
    kg_pasilla, 
    precio_pasilla_kg, 
    max_dias=90, 
    simulaciones=5000
):
    np.random.seed(42)
    retornos = np.log(df_mercado / df_mercado.shift(1)).dropna()

    media_diaria_cafe = retornos['cafe'].mean()
    media_diaria_trm = retornos['trm'].mean()

    alpha = 1 - EWMA_LAMBDA
    cov_ewma_serie = retornos.ewm(alpha=alpha, adjust=False).cov()
    ultima_fecha = retornos.index[-1]
    cov_actual = cov_ewma_serie.loc[ultima_fecha].values

    L = np.linalg.cholesky(cov_actual)

    ultimo_cafe = df_mercado['cafe'].iloc[-1]
    ultima_trm = df_mercado['trm'].iloc[-1]

    trayectorias_cafe = np.zeros((max_dias + 1, simulaciones))
    trayectorias_trm = np.zeros((max_dias + 1, simulaciones))

    trayectorias_cafe[0, :] = ultimo_cafe
    trayectorias_trm[0, :] = ultima_trm

    for t in range(1, max_dias + 1):
        Z = np.random.standard_normal(size=(simulaciones, 2))
        choques = Z @ L.T

        drift_cafe = media_diaria_cafe - 0.5 * cov_actual[0, 0]
        trayectorias_cafe[t, :] = trayectorias_cafe[t-1, :] * np.exp(drift_cafe + choques[:, 0])

        drift_trm = media_diaria_trm - 0.5 * cov_actual[1, 1]
        trayectorias_trm[t, :] = trayectorias_trm[t-1, :] * np.exp(drift_trm + choques[:, 1])

    trayectorias_carga = calcular_precio_interno_referencia(
        trayectorias_cafe, 
        trayectorias_trm, 
        diferencial_usd, 
        costos_usd, 
        libras_carga, 
        kg_pasilla, 
        precio_pasilla_kg
    )[0]

    return trayectorias_carga, trayectorias_cafe, trayectorias_trm


def calcular_todas_las_proyecciones(params: dict, df_mercado: pd.DataFrame) -> dict:
    dias_simulacion = params['dias_analisis']
    
    # 1. Ejecutar simulación
    trayectorias_carga, trayectorias_cafe, trayectorias_trm = simular_gbm_ewma_trayectorias(
        df_mercado,
        diferencial_usd=params['diferencial_usd'],
        costos_usd=params['costos_usd'],
        libras_carga=params['libras_carga'],
        kg_pasilla=params['kg_pasilla'],
        precio_pasilla_kg=params['precio_pasilla_kg'],
        max_dias=dias_simulacion,
        simulaciones=5000
    )

    precios_futuros_finales = trayectorias_carga[dias_simulacion, :]
    var_95 = np.percentile(precios_futuros_finales, 5)
    precio_promedio_sim = np.mean(precios_futuros_finales)
    trm_promedio_sim = np.mean(trayectorias_trm[dias_simulacion, :])

    opciones = params['cotizaciones'].get(params['tenor'], [])
    filas_cotizaciones = []

    for op in opciones:
        strike = op["strike"]
        prima = op["prima"]
        escenarios_reales = np.maximum(strike - precios_futuros_finales, 0.0) + precios_futuros_finales - prima
        filas_cotizaciones.append({
            "Strike (COP)": f"${strike:,.0f}",
            "Prima (COP)": f"${prima:,.0f}",
            "Piso Neto (COP)": f"${strike - prima:,.0f}",
            "Promedio Simulado": f"${np.mean(escenarios_reales):,.0f}",
            "VaR 95% Cobertura": f"${np.percentile(escenarios_reales, 5):,.0f}",
            "Costo Seguro Total": f"${(prima * params['volumen_cargas'])/1e6:,.2f} M COP"
        })

    # CLAVE AQUÍ: Asegurar que se retornen trayectorias_cafe y trayectorias_trm
    return {
        "trayectorias_carga": trayectorias_carga,
        "trayectorias_cafe": trayectorias_cafe,
        "trayectorias_trm": trayectorias_trm,
        "precios_futuros": precios_futuros_finales,
        "var_95": var_95,
        "precio_promedio_sim": precio_promedio_sim,
        "trm_promedio_sim": trm_promedio_sim,
        "df_cotizaciones": pd.DataFrame(filas_cotizaciones)
    }
