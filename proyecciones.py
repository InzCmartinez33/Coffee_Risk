import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

EWMA_LAMBDA = 0.94
COLS = ["cafe", "trm"]


# =====================================================================
# PRECIO INTERNO DE REFERENCIA (sin cambios)
# =====================================================================
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


# =====================================================================
# MÉTODO ANTERIOR (GBM gaussiano + covarianza EWMA) — se conserva como
# línea base para el backtest
# =====================================================================
def simular_gbm_ewma_trayectorias(df_mercado, max_dias=90, simulaciones=5000):
    np.random.seed(42)
    retornos = np.log(df_mercado / df_mercado.shift(1)).dropna()
    media_cafe = retornos["cafe"].mean()
    media_trm = retornos["trm"].mean()

    cov_serie = retornos.ewm(alpha=1 - EWMA_LAMBDA, adjust=False).cov()
    cov_actual = cov_serie.loc[retornos.index[-1]].values
    L = np.linalg.cholesky(cov_actual)

    tray_cafe = np.zeros((max_dias + 1, simulaciones))
    tray_trm = np.zeros((max_dias + 1, simulaciones))
    tray_cafe[0, :] = df_mercado["cafe"].iloc[-1]
    tray_trm[0, :] = df_mercado["trm"].iloc[-1]

    drift_cafe = media_cafe - 0.5 * cov_actual[0, 0]
    drift_trm = media_trm - 0.5 * cov_actual[1, 1]

    for t in range(1, max_dias + 1):
        Z = np.random.standard_normal(size=(simulaciones, 2))
        ch = Z @ L.T
        tray_cafe[t] = tray_cafe[t - 1] * np.exp(drift_cafe + ch[:, 0])
        tray_trm[t] = tray_trm[t - 1] * np.exp(drift_trm + ch[:, 1])
    return tray_cafe, tray_trm


# =====================================================================
# MÉTODO ML: volatilidad pronosticada con regresión (tipo HAR + Ridge)
# + simulación por bootstrap de residuos estandarizados (FHS)
# =====================================================================
def _ewma_vol(r, lam):
    return np.sqrt((r ** 2).ewm(alpha=1 - lam, adjust=False).mean())


def _features(r):
    f = pd.DataFrame({
        "ew94": _ewma_vol(r, 0.94),
        "ew97": _ewma_vol(r, 0.97),
        "s5": r.rolling(5).std(),
        "s21": r.rolling(21).std(),
        "s63": r.rolling(63).std(),
        "abs5": r.abs().rolling(5).mean(),
    })
    return np.log(f + 1e-8)


def _pronosticar_sigma(r, h):
    """Devuelve (sigma_ml, sigma_ewma) diarias. Predice log(vol realizada
    de los próximos h días). Sin look-ahead: solo usa filas cuyo objetivo
    ya está observado al final de `r`."""
    sigma_base = float(_ewma_vol(r, EWMA_LAMBDA).iloc[-1])
    X = _features(r)
    y = 0.5 * np.log((r ** 2).rolling(h).mean().shift(-h) + 1e-12)
    datos = pd.concat([X, y.rename("y")], axis=1).dropna()
    x_hoy = X.iloc[[-1]]

    if len(datos) < 120 or x_hoy.isna().any().any():
        return sigma_base, sigma_base  # pocos datos: cae a EWMA

    modelo = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    modelo.fit(datos[X.columns], datos["y"])
    sigma_ml = float(np.exp(modelo.predict(x_hoy)[0]))
    sigma_ml = float(np.clip(sigma_ml, 0.5 * sigma_base, 2.0 * sigma_base))
    return sigma_ml, sigma_base


def simular_gbm_ml_trayectorias(df_mercado, max_dias=90, simulaciones=5000, seed=42):
    rng = np.random.default_rng(seed)
    r = np.log(df_mercado / df_mercado.shift(1)).dropna()
    h = int(min(max_dias, 63))

    sigma = np.array([_pronosticar_sigma(r[c], h)[0] for c in COLS])

    # Residuos estandarizados por la vol EWMA del día previo. Se remuestrean
    # por fila para conservar correlación café-TRM y colas pesadas.
    z = pd.DataFrame({c: r[c] / _ewma_vol(r[c], EWMA_LAMBDA).shift(1) for c in COLS}).dropna()
    z = z.iloc[30:]                       # descarta el arranque de la EWMA (vol inicial poco fiable)
    z = (z - z.mean()) / z.std()
    z = z.clip(-6, 6)                     # evita que un solo día extremo domine las colas
    z = (z - z.mean()) / z.std()
    Z = z.values

    idx = rng.integers(0, len(Z), size=(max_dias, simulaciones))
    choques = Z[idx] * sigma                      # (dias, sims, 2)
    drift = -0.5 * sigma ** 2                     # drift ~0: la media no es predecible
    log_paths = np.cumsum(drift + choques, axis=0)
    log_paths = np.concatenate([np.zeros((1, simulaciones, 2)), log_paths], axis=0)

    ultimo = np.array([df_mercado[c].iloc[-1] for c in COLS])
    tray = ultimo * np.exp(log_paths)
    return tray[:, :, 0], tray[:, :, 1]


def simular(df_mercado, max_dias, simulaciones, metodo="ml"):
    if metodo == "ml":
        return simular_gbm_ml_trayectorias(df_mercado, max_dias, simulaciones)
    return simular_gbm_ewma_trayectorias(df_mercado, max_dias, simulaciones)


# =====================================================================
# PROYECCIONES PARA EL DASHBOARD
# =====================================================================
def calcular_todas_las_proyecciones(params: dict, df_mercado: pd.DataFrame) -> dict:
    dias = params["dias_analisis"]
    tray_cafe, tray_trm = simular(df_mercado, dias, 5000, params.get("metodo", "ml"))

    cafe_final = tray_cafe[dias, :]
    trm_final = tray_trm[dias, :]

    precios, _, _ = calcular_precio_interno_referencia(
        cafe_final, trm_final, params["diferencial_usd"], params["costos_usd"],
        params["libras_carga"], params["kg_pasilla"], params["precio_pasilla_kg"]
    )

    var_95 = np.percentile(precios, 5)
    filas = []
    for op in params["cotizaciones"].get(params["tenor"], []):
        strike, prima = op["strike"], op["prima"]
        esc = np.maximum(strike - precios, 0.0) + precios - prima
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
        "precios_futuros": precios,
        "var_95": var_95,
        "precio_promedio_sim": np.mean(precios),
        "trm_promedio_sim": np.mean(trm_final),
        "df_cotizaciones": pd.DataFrame(filas),
    }


# =====================================================================
# BACKTEST WALK-FORWARD
# =====================================================================
def backtest_walk_forward(df_mercado, horizonte, metodos=("gbm_ewma", "ml"),
                          paso=5, min_train=250, simulaciones=1500):
    """En cada fecha de origen usa SOLO datos hasta esa fecha, proyecta
    `horizonte` días y compara con el precio realmente observado."""
    filas = []
    n = len(df_mercado)
    for i in range(min_train, n - horizonte, paso):
        hist = df_mercado.iloc[: i + 1]
        real = df_mercado.iloc[i + horizonte]
        for metodo in metodos:
            tc, tt = simular(hist, horizonte, simulaciones, metodo)
            for var, tray in (("cafe", tc), ("trm", tt)):
                p5, p50, p95 = np.percentile(tray[-1], [5, 50, 95])
                filas.append({
                    "origen": hist.index[-1], "variable": var, "metodo": metodo,
                    "spot": hist[var].iloc[-1], "p5": p5, "p50": p50, "p95": p95,
                    "real": real[var],
                })
    return pd.DataFrame(filas)


def resumir_backtest(df_bt):
    filas = []
    for (var, metodo), g in df_bt.groupby(["variable", "metodo"]):
        err = (g["p50"] - g["real"]) / g["real"]
        err_rw = (g["spot"] - g["real"]) / g["real"]
        filas.append({
            "Variable": "Café NY" if var == "cafe" else "TRM",
            "Método": "GBM+EWMA (anterior)" if metodo == "gbm_ewma" else "ML (Ridge-vol + FHS)",
            "MAPE P50 (%)": err.abs().mean() * 100,
            "MAPE caminata aleatoria (%)": err_rw.abs().mean() * 100,
            "Sesgo (%)": err.mean() * 100,
            "RMSE (%)": float(np.sqrt((err ** 2).mean()) * 100),
            "Cobertura P5-P95 (%) [ideal 90]": ((g["real"] >= g["p5"]) & (g["real"] <= g["p95"])).mean() * 100,
            "Ancho intervalo (%)": ((g["p95"] - g["p5"]) / g["real"]).mean() * 100,
            "N proyecciones": len(g),
        })
    return pd.DataFrame(filas).round(2)


# =====================================================================
# BACKTEST DEL VaR DE LA CARGA (excedencias + test de Kupiec)
# =====================================================================
def backtest_var_carga(df_mercado, horizonte, params_carga,
                       metodos=("gbm_ewma", "ml"), paso=5,
                       min_train=250, simulaciones=1500, nivel=0.05):
    """En cada origen: predice el VaR (percentil `nivel`) del precio de la
    carga a `horizonte` días y lo compara con el precio de la carga que
    realmente resultó (calculado con café y TRM observados)."""
    pc = params_carga
    args = (pc["diferencial_usd"], pc["costos_usd"], pc["libras_carga"],
            pc["kg_pasilla"], pc["precio_pasilla_kg"])
    filas = []
    n = len(df_mercado)
    for i in range(min_train, n - horizonte, paso):
        hist = df_mercado.iloc[: i + 1]
        real = df_mercado.iloc[i + horizonte]
        precio_hoy = calcular_precio_interno_referencia(
            hist["cafe"].iloc[-1], hist["trm"].iloc[-1], *args)[0]
        precio_real = calcular_precio_interno_referencia(
            real["cafe"], real["trm"], *args)[0]
        for metodo in metodos:
            tc, tt = simular(hist, horizonte, simulaciones, metodo)
            precios = calcular_precio_interno_referencia(tc[-1], tt[-1], *args)[0]
            filas.append({
                "origen": hist.index[-1], "metodo": metodo,
                "precio_hoy": precio_hoy, "var": np.percentile(precios, nivel * 100),
                "precio_real": precio_real,
            })
    return pd.DataFrame(filas)


def _kupiec_pvalue(x, N, p):
    from scipy.stats import chi2
    if N == 0:
        return np.nan
    ph = x / N
    ll0 = (N - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (N - x) * np.log(1 - ph) if x < N else 0.0
    ll1 += x * np.log(ph) if x > 0 else 0.0
    return float(1 - chi2.cdf(-2 * (ll0 - ll1), df=1))


def resumir_var_carga(df_var, nivel=0.05):
    filas = []
    for metodo, g in df_var.groupby("metodo"):
        exc = g["precio_real"] < g["var"]
        x, N = int(exc.sum()), len(g)
        corto = ((g["var"] - g["precio_real"]) / g["precio_hoy"])[exc]
        filas.append({
            "Método": "GBM+EWMA (anterior)" if metodo == "gbm_ewma" else "ML (Ridge-vol + FHS)",
            "N": N,
            "Excedencias": x,
            "Tasa excedencias (%) [ideal 5]": round(100 * x / N, 2),
            "p-valor Kupiec": round(_kupiec_pvalue(x, N, nivel), 3),
            "VaR medio (% pérdida vs hoy)": round(((g["precio_hoy"] - g["var"]) / g["precio_hoy"]).mean() * 100, 2),
            "Exceso medio al fallar (% precio hoy)": round(corto.mean() * 100, 2) if x > 0 else 0.0,
            "Peor pérdida real (%)": round(((g["precio_hoy"] - g["precio_real"]) / g["precio_hoy"]).max() * 100, 2),
        })
    return pd.DataFrame(filas)
