"""
Backtest "rolling origin" de los modelos de café y TRM (y del precio de la carga).

En cada fecha de origen t el modelo se calibra SOLO con datos hasta t, simula h
días hábiles hacia adelante y se compara contra lo que realmente ocurrió en t+h.
Se usa la misma función de producción (simular_cafe_trm), así que se está
probando exactamente el modelo que alimenta el dashboard.
"""
import math
import numpy as np
import pandas as pd

from proyecciones import simular_cafe_trm, calcular_precio_interno_referencia

ETIQUETAS = {"cafe": "Café NY (USD/lb)", "trm": "TRM (COP/USD)", "carga": "Precio de la carga (COP)"}


# ---------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------
def ejecutar_backtest(df_mercado, dias_habiles, paso=5, ventana=500,
                      simulaciones=1000, params_carga=None):
    """
    df_mercado   : DataFrame con columnas 'cafe' (centavos/lb) y 'trm', índice fecha.
    dias_habiles : horizonte en días hábiles (1m≈21, 3m≈63, 6m≈126).
    paso         : cada cuántos días hábiles se toma un nuevo origen.
    ventana      : observaciones usadas para calibrar en cada origen (rolling).
    params_carga : dict opcional con diferencial_usd, costos_usd, libras_carga,
                   kg_pasilla, precio_pasilla_kg -> también se backtestea la carga.
    """
    n = len(df_mercado)
    ultimo_origen = n - 1 - dias_habiles
    if ventana > ultimo_origen or ultimo_origen - ventana < paso:
        raise ValueError("Historial insuficiente: reduce la ventana/horizonte o descarga más años de datos.")

    filas = []
    for i in range(ventana - 1, ultimo_origen + 1, paso):
        hist = df_mercado.iloc[i - ventana + 1: i + 1]
        tray_c, tray_t, _ = simular_cafe_trm(hist, dias_habiles, simulaciones=simulaciones, semilla=1000 + i)
        sim = {"cafe": tray_c[-1] / 100.0, "trm": tray_t[-1]}
        real_row = df_mercado.iloc[i + dias_habiles]
        spot_row = df_mercado.iloc[i]
        real = {"cafe": real_row["cafe"] / 100.0, "trm": real_row["trm"]}
        spot = {"cafe": spot_row["cafe"] / 100.0, "trm": spot_row["trm"]}

        if params_carga:
            args = (params_carga["diferencial_usd"], params_carga["costos_usd"],
                    params_carga["libras_carga"], params_carga["kg_pasilla"],
                    params_carga["precio_pasilla_kg"])
            sim["carga"] = calcular_precio_interno_referencia(tray_c[-1], tray_t[-1], *args)[0]
            real["carga"] = calcular_precio_interno_referencia(real_row["cafe"], real_row["trm"], *args)[0]
            spot["carga"] = calcular_precio_interno_referencia(spot_row["cafe"], spot_row["trm"], *args)[0]

        for activo, v in sim.items():
            p5, p50, p95 = np.percentile(v, [5, 50, 95])
            filas.append({
                "activo": activo,
                "fecha_origen": df_mercado.index[i],
                "fecha_objetivo": df_mercado.index[i + dias_habiles],
                "spot": spot[activo], "real": real[activo],
                "p5": p5, "p50": p50, "p95": p95,
                "pit": float((v < real[activo]).mean()),
            })
    df = pd.DataFrame(filas)
    df["bajo_p5"] = df["real"] < df["p5"]
    df["sobre_p95"] = df["real"] > df["p95"]
    return df


# ---------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------
def kupiec_pof(n, x, p=0.05):
    """Prueba de Kupiec (cobertura incondicional). Devuelve el p-valor."""
    if n == 0:
        return float("nan")
    pi = x / n
    ll0 = (n - x) * math.log(1 - p) + x * math.log(p)
    ll1 = 0.0
    if x > 0:
        ll1 += x * math.log(pi)
    if n - x > 0:
        ll1 += (n - x) * math.log(1 - pi)
    lr = max(0.0, -2.0 * (ll0 - ll1))
    return math.erfc(math.sqrt(lr / 2.0))      # chi-cuadrado, 1 g.l.


def resumen_backtest(df, dias_habiles, paso):
    filas = []
    for activo, g in df.groupby("activo"):
        n = len(g)
        x = int(g["bajo_p5"].sum())
        # Con paso < horizonte los orígenes se solapan: muestra efectiva menor.
        n_efectivo = max(1, int(n * min(1.0, paso / dias_habiles)))
        filas.append({
            "Activo": ETIQUETAS.get(activo, activo),
            "Orígenes": n,
            "Real < P5 (esperado 5%)": f"{100 * x / n:.1f}%",
            "Real > P95 (esperado 5%)": f"{100 * g['sobre_p95'].mean():.1f}%",
            "Dentro de P5–P95 (esperado 90%)": f"{100 * (1 - g['bajo_p5'].mean() - g['sobre_p95'].mean()):.1f}%",
            "Kupiec p-valor (P5)": f"{kupiec_pof(n, x):.3f}",
            "Sesgo mediana (%)": f"{100 * (g['real'] / g['p50'] - 1).mean():+.2f}",
            "Error abs. P50 (%)": f"{100 * (g['real'] / g['p50'] - 1).abs().mean():.2f}",
            "Error abs. paseo aleatorio (%)": f"{100 * (g['real'] / g['spot'] - 1).abs().mean():.2f}",
            "Obs. independientes aprox.": n_efectivo,
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------
# Gráficos interactivos (Plotly)
# ---------------------------------------------------------------------
def figura_bandas(df, activo, dias_habiles):
    import plotly.graph_objects as go
    g = df[df["activo"] == activo].sort_values("fecha_objetivo")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["fecha_objetivo"], y=g["p95"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["fecha_objetivo"], y=g["p5"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(37,99,235,0.18)", name="Banda P5–P95 del modelo",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["fecha_objetivo"], y=g["p50"], mode="lines",
                             line=dict(color="#1E3A8A", dash="dot"), name="Mediana proyectada (P50)",
                             customdata=g[["fecha_origen", "p5", "p95"]],
                             hovertemplate="Objetivo %{x|%Y-%m-%d}<br>Origen %{customdata[0]|%Y-%m-%d}"
                                           "<br>P50 %{y:,.2f}<br>P5 %{customdata[1]:,.2f} · P95 %{customdata[2]:,.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=g["fecha_objetivo"], y=g["real"], mode="lines+markers",
                             line=dict(color="#111827", width=2), marker=dict(size=5), name="Dato real",
                             hovertemplate="Real %{y:,.2f}<extra></extra>"))
    fuera = g[g["bajo_p5"] | g["sobre_p95"]]
    fig.add_trace(go.Scatter(x=fuera["fecha_objetivo"], y=fuera["real"], mode="markers",
                             marker=dict(color="#DC2626", size=10, symbol="x"), name="Fuera de la banda",
                             hovertemplate="Fuera de banda<br>Real %{y:,.2f}<extra></extra>"))
    fig.update_layout(title=f"{ETIQUETAS.get(activo, activo)} — proyección a {dias_habiles} días hábiles vs. dato real",
                      hovermode="x unified", template="plotly_white", height=430,
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=10, r=10, t=50, b=10),
                      yaxis_title=ETIQUETAS.get(activo, activo))
    return fig


def figura_pit(df, activo):
    """Histograma PIT: si el modelo está bien calibrado debe ser ~uniforme."""
    import plotly.graph_objects as go
    g = df[df["activo"] == activo]
    fig = go.Figure(go.Histogram(x=g["pit"], xbins=dict(start=0, end=1, size=0.1),
                                 marker_color="#2563EB", name="PIT"))
    fig.add_hline(y=len(g) / 10, line_dash="dash", line_color="#DC2626",
                  annotation_text="Esperado si está calibrado")
    fig.update_layout(title=f"Calibración (PIT) — {ETIQUETAS.get(activo, activo)}",
                      xaxis_title="Percentil del dato real dentro de la distribución simulada",
                      yaxis_title="Frecuencia", template="plotly_white", height=330,
                      margin=dict(l=10, r=10, t=50, b=10), bargap=0.05)
    return fig


# ---------------------------------------------------------------------
# Sección de Streamlit
# ---------------------------------------------------------------------
def render_backtest_streamlit(df_mercado, params_carga=None):
    import streamlit as st

    st.subheader("6. Backtest del modelo vs. datos reales")
    st.caption("En cada fecha pasada el modelo se calibra solo con datos anteriores, proyecta y se compara "
               "con lo que realmente ocurrió.")

    c1, c2, c3 = st.columns(3)
    horizonte = c1.selectbox("Horizonte", [("1 mes", 21), ("3 meses", 63), ("6 meses", 126)],
                             format_func=lambda x: x[0])[1]
    paso = c2.slider("Paso entre orígenes (días hábiles)", 1, 21, 5)
    ventana = c3.number_input("Ventana de calibración (obs.)", min_value=120, max_value=1500, value=500, step=50)

    if len(df_mercado) < ventana + horizonte + 30:
        st.warning(f"Solo hay {len(df_mercado)} observaciones. Descarga más historial (p. ej. 5 años) "
                   "o reduce la ventana/horizonte.")
        return

    clave = (horizonte, paso, int(ventana), len(df_mercado), str(df_mercado.index[-1]))
    if st.button("▶️ Ejecutar backtest"):
        with st.spinner("Recalibrando y simulando en cada fecha histórica..."):
            res = ejecutar_backtest(df_mercado, horizonte, paso=paso, ventana=int(ventana),
                                    params_carga=params_carga)
        st.session_state["backtest"] = (clave, res)

    guardado = st.session_state.get("backtest")
    if not guardado or guardado[0] != clave:
        st.info("Pulsa **Ejecutar backtest** para calcular (tarda unos segundos).")
        return

    res = guardado[1]
    st.dataframe(resumen_backtest(res, horizonte, paso), use_container_width=True, hide_index=True)
    st.caption("Si los orígenes se solapan (paso < horizonte), el p-valor de Kupiec es optimista: "
               "mira 'Obs. independientes aprox.'. Un p-valor bajo (< 0.05) indica que la tasa de "
               "violaciones del P5 difiere de lo esperado.")

    for activo in [a for a in ("cafe", "trm", "carga") if a in res["activo"].unique()]:
        st.plotly_chart(figura_bandas(res, activo, horizonte), use_container_width=True)
        st.plotly_chart(figura_pit(res, activo), use_container_width=True)
