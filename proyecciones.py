import pandas as pd
import numpy as np

def calcular_proyecciones(parametros: dict) -> pd.DataFrame:
    """
    Ejecuta el cálculo interno de las proyecciones.
    Retorna un DataFrame de Pandas con los resultados.
    """
    # Aquí va la lógica interna que ya tenías lista
    meses = pd.date_range(start='2026-01-01', periods=12, freq='ME')
    datos = {
        'Fecha': meses,
        'Proyeccion_Ventas': np.random.randint(100, 500, size=12),
        'Proyeccion_Costos': np.random.randint(50, 200, size=12)
    }
    df = pd.DataFrame(datos)
    return df
