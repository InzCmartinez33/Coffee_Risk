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


def simular_gbm_ewma_trayectorias(df_mercado, max_dias=90, simulaciones=5000):
    """
    Genera las trayectorias simuladas (GBM + volatilidad estructural EWMA)
    de café (KC=F) y TRM.

    Esta es la ÚNICA fuente de la proyección de mercado: tanto la sección de
    "Proyecciones Individuales" del dashboard como el motor de riesgo de la
    carga (VaR, cotizaciones de puts) consumen exactamente estos mismos
    arreglos, para que nunca queden desincronizados.
    """
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
        trayectorias_cafe[t, :] = trayectorias_cafe[t - 1, :] * np.exp(drift_cafe + choques[:, 0])

        drift_trm = media_diaria_trm - 0.5 * cov_actual[1, 1]
        trayectorias_trm[t, :] = trayectorias_trm[t - 1, :] * np.exp(drift_trm + choques[:, 1])

    return trayectorias_cafe, trayectorias_trm


def ejecutar_simulacion(dias_analisis: int, df_mercado: pd.DataFrame, simulaciones: int = 5000) -> dict:
    """
    Corre UNA sola vez la simulación GBM-EWMA y devuelve las trayectorias
    completas de café y TRM. Todo lo demás (precio de la carga, VaR,
    cotizaciones de puts) se deriva de estos mismos arreglos llamando a
    calcular_metricas_riesgo(), típicamente con los escenarios ya recortados
    al horizonte de análisis (cafe_futuro, trm_futura).
    """
    trayectorias_cafe, trayectorias_trm = simular_gbm_ewma_trayectorias(
        df_mercado, max_dias=dias_analisis, simulaciones=simulaciones
    )
    return {
        "trayectorias_cafe": trayectorias_cafe,
        "trayectorias_trm": trayectorias_trm,
    }


def calcular_metricas_riesgo(cafe_futuro: np.ndarray, trm_futura: np.ndarray, params: dict) -> dict:
    """
    Recibe los escenarios de café y TRM AL HORIZONTE de análisis (los mismos
    arreglos que el dashboard muestra en "Proyecciones Individuales") y
    calcula sobre ellos, escenario por escenario, el precio de la carga, el
    VaR 95% y las cotizaciones reales de puts.

    Al recibir cafe_futuro/trm_futura como parámetros (en vez de recalcular
    la simulación), esta función garantiza que la simulación de Montecarlo
    de la carga quede 100% amarrada a la proyección de café/TRM que ve el
    usuario en el dashboard.
    """
    precios_futuros = calcular_precio_interno_referencia(
        cafe_futuro,
        trm_futura,
        params['diferencial_usd'],
        params['costos_usd'],
        params['libras_carga'],
        params['kg_pasilla'],
        params['precio_pasilla_kg'],
    )[0]

    var_95 = np.percentile(precios_futuros, 5)
    precio_promedio_sim = np.mean(precios_futuros)
    trm_promedio_sim = np.mean(trm_futura)

    opciones = params['cotizaciones'].get(params['tenor'], [])
    filas_cotizaciones = []
    for op in opciones:
        strike = op["strike"]
        prima = op["prima"]
        escenarios_reales = np.maximum(strike - precios_futuros, 0.0) + precios_futuros - prima
        filas_cotizaciones.append({
            "Strike (COP)": f"${strike:,.0f}",
            "Prima (COP)": f"${prima:,.0f}",
            "Piso Neto (COP)": f"${strike - prima:,.0f}",
            "Promedio Simulado": f"${np.mean(escenarios_reales):,.0f}",
            "VaR 95% Cobertura": f"${np.percentile(escenarios_reales, 5):,.0f}",
            "Costo Seguro Total": f"${(prima * params['volumen_cargas']) / 1e6:,.2f} M COP",
        })

    return {
        "precios_futuros": precios_futuros,
        "var_95": var_95,
        "precio_promedio_sim": precio_promedio_sim,
        "trm_promedio_sim": trm_promedio_sim,
        "df_cotizaciones": pd.DataFrame(filas_cotizaciones),
    }
