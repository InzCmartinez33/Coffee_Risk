"""Lectura de las cotizaciones de opciones Put desde cotizaciones_put.txt"""
import os
import re
from pathlib import Path

RUTA_COTIZACIONES = Path(__file__).parent / "cotizaciones_put.txt"
TENORES_VALIDOS = ("1m", "3m", "6m")


def _numero(txt):
    limpio = re.sub(r"[\s$_]", "", txt)
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", limpio):   # 2.420.000 o 2,420,000
        limpio = re.sub(r"[.,]", "", limpio)
    return float(limpio)


def cargar_cotizaciones(ruta=RUTA_COTIZACIONES):
    """Devuelve {"1m": [{"strike":..,"prima":..}, ...], "3m": [...], "6m": [...]}.
    Lanza ValueError con el número de línea si algo está mal escrito."""
    cot = {t: [] for t in TENORES_VALIDOS}
    errores = []
    with open(ruta, encoding="utf-8") as f:
        for n, linea in enumerate(f, start=1):
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            partes = [p for p in re.split(r"[;,\t]", linea) if p.strip()]
            try:
                if len(partes) != 3:
                    raise ValueError("deben ser 3 valores: plazo, strike, prima")
                tenor = partes[0].strip().lower()
                if tenor not in TENORES_VALIDOS:
                    raise ValueError(f"plazo '{tenor}' no válido (usa 1m, 3m o 6m)")
                strike, prima = _numero(partes[1]), _numero(partes[2])
                if strike <= 0 or prima < 0:
                    raise ValueError("strike debe ser > 0 y prima >= 0")
                cot[tenor].append({"strike": strike, "prima": prima})
            except ValueError as e:
                errores.append(f"línea {n}: {e}  -> '{linea}'")
    if errores:
        raise ValueError("Errores en cotizaciones_put.txt:\n" + "\n".join(errores))
    return cot


def fecha_modificacion(ruta=RUTA_COTIZACIONES):
    return os.path.getmtime(ruta)
