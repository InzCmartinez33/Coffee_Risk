"""
Backtest "rolling origin" de los modelos de café y TRM (y del precio de la carga).

En cada fecha de origen t el modelo se calibra SOLO con datos hasta t, simula h
días hábiles hacia adelante y se compara contra lo que realmente ocurrió en t+h.
Se usa la misma función de producción (simular_cafe_trm).

Se guardan varios percentiles del pronóstico; la banda que se grafica y evalúa
(por defecto P20–P80) se elige después, sin volver a simular.
"""
import math
import numpy as np
import pandas as pd

from proyecciones import simular_cafe_trm, calcular_precio_interno_referencia

ETIQUETAS = {"cafe": "Café NY (USD/lb)", "trm": "TRM (COP/USD)", "carga": "Precio de la carga (COP)"}
QS = [5, 10, 20, 25, 50, 75, 80, 90, 95]          # percentiles almacenados
BAJOS = [5, 10, 20, 25, 50]                         # opciones para el límite inferior
ALTOS = [50, 75, 80, 90, 95]                        # opciones para el límite superior


# ---------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------
def ejecutar_backtest(df_mercado, dias_habiles, paso=5, ventana=500,
                      simulaciones=1000, params_carga=None):
    """
    df_mercado   : DataFrame con 'cafe' (centavos/lb) y 'trm', índice fecha.
    dias_habiles : horizonte en días hábiles (1m≈21, 3m≈63, 6m≈126).
    paso         : cada cuántos días hábiles se toma un nuevo origen.
    ventana      : observaciones usadas para calibrar en cada origen (rolling).
    params_carga : dict opcional (diferencial_usd, costos_usd, libras_carga,
                   kg_pasilla, precio_pasilla_kg) para backtestear también la carga.
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
        real_row, spot_row = df_mercado.iloc[i + dias_habiles], df_mercado.iloc[i]
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
            fila = {"activo": activo,
                    "fecha_origen": df_mercado.index[i],
                    "fecha_objetivo": df_mercado.index[i + dias_habiles],
                    "spot": spot[activo], "real": real[activo],
                    "pit": float((v < real[activo]).mean())}
            for q, val in zip(QS, np.percentile(v, QS)):
                fila[f"p{q}"] = val
            filas.append(fila)
    return pd.DataFrame(filas)


def _marcar(df, q_bajo, q_alto):
    """Agrega los límites elegidos y si el real quedó fuera de la banda."""
    if q_bajo not in QS or q_alto not in QS or q_bajo >= q_alto:
        raise ValueError(f"Banda inválida: P{q_bajo}–P{q_alto}")
    d = df.copy()
    d["lim_bajo"], d["lim_alto"] = d[f"p{q_bajo}"], d[f"p{q_alto}"]
    d["bajo"] = d["real"] < d["lim_bajo"]
    d["sobre"] = d["real"] > d["lim_alto"]
    return d


# ---------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------
def kupiec_pof(n, x, p):
    """Kupiec (cobertura incondicional): x violaciones en n, esperadas con prob. p. Devuelve p-valor."""
    if n == 0:
        return float("nan")
    pi = x / n
    ll0 = (n - x) * math.log(1 - p) + x * math.log(p)
    ll1 = (x * math.log(pi) if x > 0 else 0.0) + ((n - x) * math.log(1 - pi) if n - x > 0 else 0.0)
    lr = max(0.0, -2.0 * (ll0 - ll1))
    return math.erfc(math.sqrt(lr / 2.0))          # chi-cuadrado, 1 g.l.


def resumen_backtest(df, dias_habiles, paso, q_bajo=20, q_alto=80):
    d = _marcar(df, q_bajo, q_alto)
    nominal = q_alto - q_bajo
    filas = []
    for activo, g in d.groupby("activo"):
        n = len(g)
        fuera = int((g["bajo"] | g["sobre"]).sum())
        n_efectivo = max(1, int(n * min(1.0, paso / dias_habiles)))   # orígenes solapados
        filas.append({
            "Activo": ETIQUETAS.get(activo, activo),
            "Orígenes": n,
            f"Dentro de P{q_bajo}–P{q_alto} (esperado {nominal}%)": f"{100 * (1 - fuera / n):.1f}%",
            f"Real < P{q_bajo} (esperado {q_bajo}%)": f"{100 * g['bajo'].mean():.1f}%",
            f"Real > P{q_alto} (esperado {100 - q_alto}%)": f"{100 * g['sobre'].mean():.1f}%",
            "Kupiec p-valor (banda)": f"{kupiec_pof(n, fuera, 1 - nominal / 100):.3f}",
            "Real < P5 — VaR 95% (esperado 5%)": f"{100 * (g['real'] < g['p5']).mean():.1f}%",
            "Sesgo mediana (%)": f"{100 * (g['real'] / g['p50'] - 1).mean():+.2f}",
            "Error abs. P50 (%)": f"{100 * (g['real'] / g['p50'] - 1).abs().mean():.2f}",
            "Error abs. paseo aleatorio (%)": f"{100 * (g['real'] / g['spot'] - 1).abs().mean():.2f}",
            "Obs. independientes aprox.": n_efectivo,
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------
# Gráficos interactivos (Plotly)
# ---------------------------------------------------------------------
def figura_bandas(df, activo, dias_habiles, q_bajo=20, q_alto=80):
    import plotly.graph_objects as go
    g = _marcar(df, q_bajo, q_alto)
    g = g[g["activo"] == activo].sort_values("fecha_objetivo")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["fecha_objetivo"], y=g["lim_alto"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["fecha_objetivo"], y=g["lim_bajo"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(37,99,235,0.22)",
                             name=f"Banda P{q_bajo}–P{q_alto}", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=g["fecha_objetivo"], y=g["p50"], mode="lines",
                             line=dict(color="#1E3A8A", dash="dot"), name="Mediana proyectada (P50)",
                             customdata=g[["fecha_origen", "lim_bajo", "lim_alto"]],
                             hovertemplate="Objetivo %{x|%Y-%m-%d}<br>Origen %{customdata[0]|%Y-%m-%d}"
                                           f"<br>P50 %{{y:,.2f}}<br>P{q_bajo} %{{customdata[1]:,.2f}} · "
                                           f"P{q_alto} %{{customdata[2]:,.2f}}<extra></extra>"))
    fig.add_trace(go.Scatter(x=g["fecha_objetivo"], y=g["real"], mode="lines+markers",
                             line=dict(color="#111827", width=2), marker=dict(size=5), name="Dato real",
                             hovertemplate="Real %{y:,.2f}<extra></extra>"))
    fuera = g[g["bajo"] | g["sobre"]]
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


def tabla_real_vs_p50(df, activo, q_bajo=20, q_alto=80):
    """Una fila por fecha objetivo, con el error del P50 y la banda elegida."""
    g = _marcar(df, q_bajo, q_alto)
    g = g[g["activo"] == activo].sort_values("fecha_objetivo")
    out = pd.DataFrame({
        "Fecha origen": g["fecha_origen"].dt.date,
        "Fecha objetivo": g["fecha_objetivo"].dt.date,
        "Precio al origen": g["spot"],
        "P50 proyectado": g["p50"],
        "Real": g["real"],
        "Error (Real − P50)": g["real"] - g["p50"],
        "Error %": 100 * (g["real"] / g["p50"] - 1),
        f"P{q_bajo}": g["lim_bajo"],
        f"P{q_alto}": g["lim_alto"],
        "Resultado": np.where(g["bajo"], f"Bajo P{q_bajo}", np.where(g["sobre"], f"Sobre P{q_alto}", "Dentro")),
    })
    return out.reset_index(drop=True)


def figura_real_vs_p50(df, activo):
    """Izquierda: P50 (x) vs real (y) con la diagonal ideal. Derecha: error % por fecha."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    g = df[df["activo"] == activo].sort_values("fecha_objetivo")
    err = 100 * (g["real"] / g["p50"] - 1)
    lo, hi = float(min(g["p50"].min(), g["real"].min())), float(max(g["p50"].max(), g["real"].max()))

    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.1,
                        subplot_titles=("Real vs. P50 (ideal: sobre la diagonal)", "Error % del P50 por fecha"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#6B7280", dash="dash"),
                             name="Predicción perfecta", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=g["p50"], y=g["real"], mode="markers", name="Fechas",
        marker=dict(size=8, color=err, colorscale="RdBu", cmid=0, showscale=True,
                    colorbar=dict(title="Error %", x=0.43, len=0.8)),
        customdata=np.stack([g["fecha_objetivo"].dt.strftime("%Y-%m-%d"), err], axis=1),
        hovertemplate="Objetivo %{customdata[0]}<br>P50 %{x:,.2f}<br>Real %{y:,.2f}"
                      "<br>Error %{customdata[1]:.2f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=g["fecha_objetivo"], y=err, name="Error %",
                         marker_color=np.where(err >= 0, "#2563EB", "#DC2626"),
                         hovertemplate="%{x|%Y-%m-%d}<br>Error %{y:.2f}%<extra></extra>"), row=1, col=2)
    fig.add_hline(y=0, line_color="#111827", row=1, col=2)
    fig.update_xaxes(title_text="P50 proyectado", row=1, col=1)
    fig.update_yaxes(title_text="Valor real", row=1, col=1)
    fig.update_yaxes(title_text="(Real / P50 − 1) %", row=1, col=2)
    fig.update_layout(template="plotly_white", height=420, showlegend=False,
                      margin=dict(l=10, r=10, t=60, b=10),
                      title=f"{ETIQUETAS.get(activo, activo)} — valor real contra P50 de la proyección")
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

    b1, b2 = st.columns(2)
    q_bajo = b1.selectbox("Límite inferior de la banda", BAJOS, index=BAJOS.index(20),
                          format_func=lambda q: f"P{q}")
    q_alto = b2.selectbox("Límite superior de la banda", ALTOS, index=ALTOS.index(80),
                          format_func=lambda q: f"P{q}")
    if q_bajo >= q_alto:
        st.warning("El límite inferior debe ser menor que el superior.")
        return

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
        st.info("Pulsa **Ejecutar backtest** para calcular (tarda unos segundos). "
                "La banda se puede cambiar después sin volver a ejecutar.")
        return

    res = guardado[1]
    st.dataframe(resumen_backtest(res, horizonte, paso, q_bajo, q_alto), use_container_width=True, hide_index=True)
    st.caption("Una banda más angosta no reduce la incertidumbre del mercado: solo cambia qué porcentaje de "
               f"casos se espera que caiga dentro (aquí {q_alto - q_bajo}%). Un modelo calibrado debe acertar "
               "ese porcentaje, no más ni menos. Si los orígenes se solapan (paso < horizonte), el p-valor de "
               "Kupiec es optimista: mira 'Obs. independientes aprox.'.")

    activos = [a for a in ("cafe", "trm", "carga") if a in res["activo"].unique()]
    for tab, activo in zip(st.tabs([ETIQUETAS[a] for a in activos]), activos):
        with tab:
            st.plotly_chart(figura_real_vs_p50(res, activo), use_container_width=True)
            tabla = tabla_real_vs_p50(res, activo, q_bajo, q_alto)
            st.markdown("**Detalle: real vs. P50 por fecha**")
            st.dataframe(
                tabla, use_container_width=True, hide_index=True,
                column_config={c: st.column_config.NumberColumn(format="%.2f")
                               for c in ("Precio al origen", "P50 proyectado", "Real", "Error (Real − P50)",
                                         f"P{q_bajo}", f"P{q_alto}")}
                | {"Error %": st.column_config.NumberColumn(format="%+.2f%%")},
            )
            st.download_button("⬇️ Descargar tabla (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                               file_name=f"backtest_{activo}_{horizonte}d.csv", mime="text/csv",
                               key=f"dl_{activo}")
            st.plotly_chart(figura_bandas(res, activo, horizonte, q_bajo, q_alto), use_container_width=True)
            st.plotly_chart(figura_pit(res, activo), use_container_width=True)
