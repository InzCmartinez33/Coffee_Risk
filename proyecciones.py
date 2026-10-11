import numpy as np
import pandas as pd

DIAS_HABILES_ANIO = 252
DIAS_CALENDARIO_ANIO = 365

# =====================================================================
# PARÁMETROS DE CADA MODELO
#   alpha / beta = None  -> se calibran por máxima verosimilitud (GARCH(1,1)
#                           con "variance targeting") sobre el historial.
#   drift_anual          -> deriva esperada anual del log-precio.
#                           Café: 0 (el precio de un futuro es martingala).
#                           TRM : 0 (conservador para VaR; los puntos forward
#                           pertenecen al COSTO de la cobertura, no a la
#                           proyección de riesgo).
#   vida_media_dias      -> si se indica, la TRM revierte a su media de largo
#                           plazo (Ornstein-Uhlenbeck sobre log-TRM).
#                           None = sin reversión (recomendado con < 5 años de datos).
# =====================================================================
PARAMS_CAFE = {"alpha": None, "beta": None, "drift_anual": 0.0}
PARAMS_TRM = {"alpha": None, "beta": None, "drift_anual": 0.0,
              "vida_media_dias": None, "theta_log": None}


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


# ---------------------------------------------------------------------
# Calibración de volatilidad (GARCH(1,1) con variance targeting)
# ---------------------------------------------------------------------
def _calibrar_garch(r):
    """Busca (alpha, beta) en una malla maximizando la verosimilitud gaussiana."""
    var_lr = r.var()
    A, B = np.meshgrid(np.arange(0.02, 0.16, 0.01), np.arange(0.78, 0.975, 0.01))
    A, B = A.ravel(), B.ravel()
    ok = (A + B) < 0.995
    A, B = A[ok], B[ok]
    omega = (1.0 - A - B) * var_lr
    var = np.full(A.shape, var_lr)
    ll = np.zeros(A.shape)
    for t in range(1, len(r)):
        var = omega + A * r[t - 1] ** 2 + B * var
        ll += -0.5 * (np.log(var) + r[t] ** 2 / var)
    k = int(np.argmax(ll))
    return float(A[k]), float(B[k])


def _calibrar_modelo(r_log, params):
    """Devuelve residuos estandarizados (para bootstrap) y estado de varianza actual."""
    r = r_log - r_log.mean()
    var_lr = r.var()
    alpha, beta = params.get("alpha"), params.get("beta")
    if alpha is None or beta is None:
        alpha, beta = _calibrar_garch(r)
    omega = (1.0 - alpha - beta) * var_lr

    var = np.empty(len(r))
    var[0] = var_lr
    for t in range(1, len(r)):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]

    z = r / np.sqrt(var)
    z = (z - z.mean()) / z.std()                      # media 0, varianza 1, colas gruesas intactas
    var_next = omega + alpha * r[-1] ** 2 + beta * var[-1]
    return {"z": z, "omega": omega, "alpha": alpha, "beta": beta,
            "var0": var_next, "vol_anual_actual": float(np.sqrt(var_next * DIAS_HABILES_ANIO))}


# ---------------------------------------------------------------------
# Simulación: dos modelos distintos, un solo bootstrap conjunto
# ---------------------------------------------------------------------
def simular_cafe_trm(df_mercado, dias_habiles, simulaciones=5000,
                     params_cafe=None, params_trm=None, semilla=42):
    """
    Café y TRM tienen su propia dinámica (volatilidad, deriva, reversión), pero
    cada día simulado se sortea UNA fecha histórica y se toman de ella los
    residuos de ambos activos. Así se conserva la correlación (y la dependencia
    en las colas) empírica sin imponer normalidad ni Cholesky.
    """
    pc = {**PARAMS_CAFE, **(params_cafe or {})}
    pt = {**PARAMS_TRM, **(params_trm or {})}
    rng = np.random.default_rng(semilla)

    ret = np.log(df_mercado[["cafe", "trm"]] / df_mercado[["cafe", "trm"]].shift(1)).dropna()
    mc = _calibrar_modelo(ret["cafe"].to_numpy(), pc)
    mt = _calibrar_modelo(ret["trm"].to_numpy(), pt)

    idx = rng.integers(0, len(ret), size=(dias_habiles, simulaciones))

    x_c = np.full(simulaciones, np.log(df_mercado["cafe"].iloc[-1]))
    x_t = np.full(simulaciones, np.log(df_mercado["trm"].iloc[-1]))
    var_c = np.full(simulaciones, mc["var0"])
    var_t = np.full(simulaciones, mt["var0"])

    mu_c = pc["drift_anual"] / DIAS_HABILES_ANIO
    mu_t = pt["drift_anual"] / DIAS_HABILES_ANIO
    vm = pt.get("vida_media_dias")
    kappa = np.log(2.0) / vm if vm else 0.0
    theta = pt.get("theta_log")
    if theta is None:
        theta = np.log(df_mercado["trm"]).mean()

    tray_c = np.empty((dias_habiles + 1, simulaciones))
    tray_t = np.empty((dias_habiles + 1, simulaciones))
    tray_c[0], tray_t[0] = np.exp(x_c), np.exp(x_t)

    for t in range(dias_habiles):
        sh_c = np.sqrt(var_c) * mc["z"][idx[t]]
        sh_t = np.sqrt(var_t) * mt["z"][idx[t]]

        x_c = x_c + mu_c - 0.5 * var_c + sh_c
        x_t = x_t + mu_t - 0.5 * var_t + kappa * (theta - x_t) + sh_t

        var_c = mc["omega"] + mc["alpha"] * sh_c ** 2 + mc["beta"] * var_c
        var_t = mt["omega"] + mt["alpha"] * sh_t ** 2 + mt["beta"] * var_t

        tray_c[t + 1], tray_t[t + 1] = np.exp(x_c), np.exp(x_t)

    info = {"cafe": {k: mc[k] for k in ("alpha", "beta", "vol_anual_actual")},
            "trm": {k: mt[k] for k in ("alpha", "beta", "vol_anual_actual")}}
    return tray_c, tray_t, info


def calcular_todas_las_proyecciones(params: dict, df_mercado: pd.DataFrame) -> dict:
    dias_cal = params["dias_analisis"]
    # Los retornos son por día HÁBIL: se convierte el horizonte calendario.
    dias_habiles = max(1, int(round(dias_cal * DIAS_HABILES_ANIO / DIAS_CALENDARIO_ANIO)))

    tray_cafe, tray_trm, info_modelos = simular_cafe_trm(
        df_mercado, dias_habiles, simulaciones=5000,
        params_cafe=params.get("params_cafe"), params_trm=params.get("params_trm"),
    )

    cafe_final = tray_cafe[-1, :]
    trm_final = tray_trm[-1, :]

    precios_futuros_finales, _, _ = calcular_precio_interno_referencia(
        cafe_final, trm_final,
        params["diferencial_usd"], params["costos_usd"],
        params["libras_carga"], params["kg_pasilla"], params["precio_pasilla_kg"],
    )

    var_95 = np.percentile(precios_futuros_finales, 5)
    precio_promedio_sim = np.mean(precios_futuros_finales)
    trm_promedio_sim = np.mean(trm_final)

    filas = []
    for op in params["cotizaciones"].get(params["tenor"], []):
        strike, prima = op["strike"], op["prima"]
        esc = np.maximum(strike - precios_futuros_finales, 0.0) + precios_futuros_finales - prima
        filas.append({
            "Strike (COP)": f"${strike:,.0f}",
            "Prima (COP)": f"${prima:,.0f}",
            "Piso Neto (COP)": f"${strike - prima:,.0f}",
            "Promedio Simulado": f"${np.mean(esc):,.0f}",
            "VaR 95% Cobertura": f"${np.percentile(esc, 5):,.0f}",
            "Costo Seguro Total": f"${(prima * params['volumen_cargas']) / 1e6:,.2f} M COP",
        })

    return {
        "trayectorias_cafe": tray_cafe,
        "trayectorias_trm": tray_trm,
        "cafe_final": cafe_final,
        "trm_final": trm_final,
        "dias_habiles": dias_habiles,
        "bandas_cafe": np.percentile(tray_cafe, [5, 50, 95], axis=1),
        "bandas_trm": np.percentile(tray_trm, [5, 50, 95], axis=1),
        "info_modelos": info_modelos,
        "precios_futuros": precios_futuros_finales,
        "var_95": var_95,
        "precio_promedio_sim": precio_promedio_sim,
        "trm_promedio_sim": trm_promedio_sim,
        "df_cotizaciones": pd.DataFrame(filas),
    }
