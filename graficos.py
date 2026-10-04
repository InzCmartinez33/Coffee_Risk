import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def grafico_proyeccion(serie_hist, trayectorias, titulo, ylabel, color,
                       factor=1.0, dias_hist=180, formato="{x:,.2f}"):
    """Precio real (histórico) + proyección con bandas de probabilidad.

    serie_hist:   pd.Series con índice de fechas (precio real)
    trayectorias: array (dias+1, simulaciones) de la simulación
    factor:       divisor de unidades (100 para pasar centavos a USD/lb)
    """
    hist = serie_hist.iloc[-dias_hist:] / factor
    ultima = serie_hist.index[-1]
    dias = trayectorias.shape[0] - 1
    fechas_f = pd.date_range(ultima, periods=dias + 1, freq="D")
    p5, p25, p50, p75, p95 = np.percentile(trayectorias / factor, [5, 25, 50, 75, 95], axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(hist.index, hist.values, color=color, lw=1.8, label="Precio real")
    ax.fill_between(fechas_f, p5, p95, color=color, alpha=0.15, label="Rango P5–P95")
    ax.fill_between(fechas_f, p25, p75, color=color, alpha=0.30, label="Rango P25–P75")
    ax.plot(fechas_f, p50, color="#1E3A8A", ls="--", lw=1.8, label="Proyección (P50)")
    ax.axvline(ultima, color="gray", ls=":", lw=1)

    for y, txt in ((p95[-1], "P95"), (p50[-1], "P50"), (p5[-1], "P5")):
        ax.annotate(f"{txt}: {formato.format(x=y)}", xy=(fechas_f[-1], y),
                    xytext=(4, 0), textcoords="offset points", fontsize=7, va="center")

    ax.set_title(titulo, fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.yaxis.set_major_formatter(formato)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.tick_params(labelsize=7)
    ax.set_xlim(hist.index[0], fechas_f[-1] + (fechas_f[-1] - hist.index[0]) * 0.12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig
