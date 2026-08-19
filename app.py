import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests

# Configuración inicial de la página
st.set_page_config(
    page_title="Simulador P&G y TRM Café",
    page_icon="☕",
    layout="wide"
)

# ==============================================================================
# 1. PARÁMETROS CORE FNC Y CONSULTAS EN VIVO DE TRM
# ==============================================================================
FACTOR_RENDIMIENTO_BASE = 92.8
PRIMA_COLOMBIA_LBS = 0.15
DESCUENTO_PASILLA_COP = 25000

@st.cache_data(ttl=1800)  # Guarda los datos en caché por 30 minutos
def obtener_historico_trm(dias_atras=30):
    """
    Obtiene la TRM actual y consulta los últimos N días desde la API de Datos Abiertos
    para calcular Mínimos, Máximos y Promedios Históricos.
    """
    url = f"https://www.datos.gov.co/resource/32sa-213a.json?$order=vigenciasta%20DESC&$limit={dias_atras}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        df['valor'] = df['valor'].astype(float)
        
        trm_actual = df.iloc[0]['valor']
        trm_min_hist = df['valor'].min()
        trm_max_hist = df['valor'].max()
        trm_prom_hist = df['valor'].mean()
        
        return {
            'actual': trm_actual,
            'min': trm_min_hist,
            'max': trm_max_hist,
            'prom': trm_prom_hist,
            'status': True
        }
    except Exception as e:
        # Fallback de seguridad en caso de fallo de conexión/API
        return {
            'actual': 3950.0,
            'min': 3800.0,
            'max': 4100.0,
            'prom': 3950.0,
            'status': False
        }

def calcular_precio_carga_fnc(cafe_ny_cents, trm, factor_rendimiento=92.8, prima_colombia=0.15):
    """Fórmula oficial FNC para calcular el precio por Carga de 125 kg."""
    cafe_ny_usd = cafe_ny_cents / 100.0
    precio_fob_usd_lb = cafe_ny_usd + prima_colombia
    libras_exportables = (125.0 / factor_rendimiento) * 70.0 * 2.20462
    valor_exportable_cop = libras_exportables * precio_fob_usd_lb * trm
    return valor_exportable_cop + DESCUENTO_PASILLA_COP

# ==============================================================================
# 2. BARRA LATERAL (ENTRADAS DE DATOS E HISTÓRICO TRM)
# ==============================================================================
st.sidebar.header("⚙️ Configuración del Mercado")

# Cargar histórico de TRM automáticamente
dias_analisis = st.sidebar.slider("Días de histórico TRM a consultar", 7, 90, 30)
trm_info = obtener_historico_trm(dias_analisis)

if trm_info['status']:
    st.sidebar.success(f"✅ TRM Hoy: **${trm_info['actual']:,.2f} COP**")
else:
    st.sidebar.warning("⚠️ Usando valores fallback para TRM")

# Parámetros del Proyector de TRM trayendo Máximos y Mínimos automáticos
st.sidebar.subheader("📌 Rangos de Simulación TRM")
trm_min = st.sidebar.number_input("TRM Mínima ($)", value=float(np.floor(trm_info['min'])))
trm_max = st.sidebar.number_input("TRM Máxima ($)", value=float(np.ceil(trm_info['max'])))

st.sidebar.caption(f"Histórico ({dias_analisis}d): Mín: ${trm_info['min']:,.0f} | Máx: ${trm_info['max']:,.0f}")

st.sidebar.subheader("☕ Precio Bolsa NY (c/lb)")
cafe_min = st.sidebar.number_input("Bolsa NY Mínimo", value=210.0)
cafe_max = st.sidebar.number_input("Bolsa NY Máximo", value=260.0)

st.sidebar.subheader("📦 Volumen y Coberturas Banco")
cargas_a_vender = st.sidebar.number_input("Cargas a comercializar (125kg)", value=100, step=10)

fwd_30 = st.sidebar.number_input("TRM Forward 30D ($)", value=round(trm_info['actual'] + 30, 0))
costo_30 = st.sidebar.number_input("Costo Cobertura 30D ($)", value=15.0)

fwd_60 = st.sidebar.number_input("TRM Forward 60D ($)", value=round(trm_info['actual'] + 65, 0))
costo_60 = st.sidebar.number_input("Costo Cobertura 60D ($)", value=28.0)

fwd_90 = st.sidebar.number_input("TRM Forward 90D ($)", value=round(trm_info['actual'] + 100, 0))
costo_90 = st.sidebar.number_input("Costo Cobertura 90D ($)", value=42.0)

# ==============================================================================
# 3. CUERPO PRINCIPAL Y MOTOR MONTE CARLO
# ==============================================================================
st.title("☕ Simulador de Riesgo y P&G Cafetero")
st.markdown("Herramienta interactiva para la proyección de ingresos y análisis de coberturas cambarias en Colombia.")

if st.button("🚀 Ejecutar Simulaciones Monte Carlo", type="primary"):
    simulaciones = 10000
    
    # A. Generación de matriz de Monte Carlo usando límites ajustados
    trm_sim = np.random.triangular(trm_min, (trm_min + trm_max)/2, trm_max, size=simulaciones)
    cafe_sim = np.random.triangular(cafe_min, (cafe_min + cafe_max)/2, cafe_max, size=simulaciones)
    
    carga_sim = calcular_precio_carga_fnc(cafe_sim, trm_sim)
    
    p5_unit = np.percentile(carga_sim, 5)
    p50_unit = np.percentile(carga_sim, 50)
    cafe_mediana = (cafe_min + cafe_max) / 2
    
    # B. Cálculos de Forwards
    datos_forwards = {
        "30 días": {"fwd": fwd_30, "costo": costo_30},
        "60 días": {"fwd": fwd_60, "costo": costo_60},
        "90 días": {"fwd": fwd_90, "costo": costo_90}
    }
    
    fwds_calc = {}
    for plazo, datos in datos_forwards.items():
        trm_efec = datos["fwd"] - datos["costo"]
        bruto_u = calcular_precio_carga_fnc(cafe_mediana, datos["fwd"])
        
        libras_exp = (125.0 / FACTOR_RENDIMIENTO_BASE) * 70.0 * 2.20462
        precio_fob_usd = (cafe_mediana / 100.0) + PRIMA_COLOMBIA_LBS
        costo_c_u = libras_exp * precio_fob_usd * datos["costo"]
        neto_u = bruto_u - costo_c_u
        
        fwds_calc[plazo] = {
            'trm_efectiva': trm_efec,
            'carga_neto': neto_u,
            'ingreso_bruto_total': bruto_u * cargas_a_vender,
            'costo_total': costo_c_u * cargas_a_vender,
            'ingreso_neto_total': neto_u * cargas_a_vender
        }

    ingreso_spot_estres = p5_unit * cargas_a_vender
    ingreso_spot_mediana = p50_unit * cargas_a_vender

    # C. Renderizado del Dashboard de Matplotlib
    fig = plt.figure(figsize=(15, 10), facecolor='#F8FAFC')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.1, 1], width_ratios=[1.15, 1])
    colores_fwd = {'30 días': '#2563EB', '60 días': '#7C3AED', '90 días': '#D97706'}
    
    # Cuadrante 1: Histograma
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(carga_sim * cargas_a_vender / 1e6, bins=45, color='#94A3B8', alpha=0.5, edgecolor='white')
    ax1.axvline(ingreso_spot_estres / 1e6, color='#EF4444', linestyle=':', label=f'P5 Estrés (${ingreso_spot_estres:,.0f})')
    ax1.axvline(ingreso_spot_mediana / 1e6, color='#10B981', linestyle='--', label=f'P50 Mediana (${ingreso_spot_mediana:,.0f})')
    for p, info in fwds_calc.items():
        ax1.axvline(info['ingreso_neto_total'] / 1e6, color=colores_fwd[p], label=f'Fwd {p} Neto')
    ax1.set_title(f"1. Distribución Monte Carlo ({cargas_a_vender:,} Cargas)", fontweight='bold')
    ax1.set_xlabel("Ingresos Totales (Millones COP)")
    ax1.legend(fontsize=8)

    # Cuadrante 2: Mapa de Sensibilidad (RESTAURADO)
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(cafe_sim, trm_sim, c=carga_sim/1e6, cmap='YlGnBu', alpha=0.35, s=10)
    cbar = fig.colorbar(scatter, ax=ax2)
    cbar.set_label('Precio Carga Spot (M COP)', rotation=270, labelpad=15)
    for p, info in fwds_calc.items():
        ax2.axhline(info['trm_efectiva'], color=colores_fwd[p], linestyle='--', label=f'TRM Efec. {p} (${info["trm_efectiva"]:,.0f})')
    ax2.set_title("2. Mapa de Sensibilidad Mercado Spot vs TRM Pactada", fontweight='bold')
    ax2.set_xlabel("Bolsa NY (Cents / lb)")
    ax2.set_ylabel("TRM ($ COP)")
    ax2.legend(fontsize=8)

    # Cuadrante 3: Tabla P&G Gráfica
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')
    
    columnas = ['Concepto', 'Spot (P50)', 'Fwd 30D', 'Fwd 60D', 'Fwd 90D']
    filas = [
        ['TRM Efec.', f"${trm_info['actual']:,.0f}", f"${fwds_calc['30 días']['trm_efectiva']:,.0f}", f"${fwds_calc['60 días']['trm_efectiva']:,.0f}", f"${fwds_calc['90 días']['trm_efectiva']:,.0f}"],
        ['$/Carga Neto', f"${p50_unit:,.0f}", f"${fwds_calc['30 días']['carga_neto']:,.0f}", f"${fwds_calc['60 días']['carga_neto']:,.0f}", f"${fwds_calc['90 días']['carga_neto']:,.0f}"],
        ['Ing. Bruto', f"${ingreso_spot_mediana:,.0f}", f"${fwds_calc['30 días']['ingreso_bruto_total']:,.0f}", f"${fwds_calc['60 días']['ingreso_bruto_total']:,.0f}", f"${fwds_calc['90 días']['ingreso_bruto_total']:,.0f}"],
        ['Costo Cobertura', "$0", f"-${fwds_calc['30 días']['costo_total']:,.0f}", f"-${fwds_calc['60 días']['costo_total']:,.0f}", f"-${fwds_calc['90 días']['costo_total']:,.0f}"],
        ['ING. NETO P&G', f"${ingreso_spot_mediana:,.0f}", f"${fwds_calc['30 días']['ingreso_neto_total']:,.0f}", f"${fwds_calc['60 días']['ingreso_neto_total']:,.0f}", f"${fwds_calc['90 días']['ingreso_neto_total']:,.0f}"],
        ['Dif. vs Spot', "$0", f"${fwds_calc['30 días']['ingreso_neto_total'] - ingreso_spot_mediana:,.0f}", f"${fwds_calc['60 días']['ingreso_neto_total'] - ingreso_spot_mediana:,.0f}", f"${fwds_calc['90 días']['ingreso_neto_total'] - ingreso_spot_mediana:,.0f}"]
    ]
    
    tabla = ax3.table(cellText=filas, colLabels=columnas, loc='center', cellLoc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(8)
    tabla.scale(1, 1.3)
    
    for (i, j), cell in tabla.get_celld().items():
        if i == 0:
            cell.set_facecolor('#1E293B')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
        elif i == 4:
            cell.set_facecolor('#E0F2FE')
            cell.get_text().set_weight('bold')

    ax3.set_title("3. Estado de Pérdidas y Ganancias (P&G) Comparativo", fontweight='bold')

    # Cuadrante 4: Cuadro de Control y Riesgo
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    texto_resumen = (
        f"📊 RESUMEN DE COBERTURA Y PROTECCIÓN\n"
        f"_________________________________________________\n\n"
        f"• TRM Promedio Últimos {dias_analisis} Días: ${trm_info['prom']:,.2f} COP\n"
        f"• Piso de Protección Estrés (P5 Spot): ${ingreso_spot_estres:,.0f} COP\n\n"
        f"🛡️ CAPITAL PROTEGIDO CONTRA CAÍDAS:\n"
        f"  - Con Forward 30D aseguras: +${fwds_calc['30 días']['ingreso_neto_total'] - ingreso_spot_estres:,.0f} COP adicionales\n"
        f"  - Con Forward 60D aseguras: +${fwds_calc['60 días']['ingreso_neto_total'] - ingreso_spot_estres:,.0f} COP adicionales\n"
        f"  - Con Forward 90D aseguras: +${fwds_calc['90 días']['ingreso_neto_total'] - ingreso_spot_estres:,.0f} COP adicionales"
    )
    
    ax4.text(0.05, 0.95, texto_resumen, transform=ax4.transAxes, fontsize=9.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='#F1F5F9', edgecolor='#CBD5E1'))

    plt.tight_layout()
    st.pyplot(fig)

    # Tarjetas Métricas Rápidas
    c1, c2, c3 = st.columns(3)
    c1.metric("Ingreso Esperado (Spot Mediana)", f"${ingreso_spot_mediana:,.0f} COP")
    c2.metric("Ingreso Garantizado (Fwd 90D)", f"${fwds_calc['90 días']['ingreso_neto_total']:,.0f} COP")
    c3.metric("Capital Protegido vs Estrés", f"+${fwds_calc['90 días']['ingreso_neto_total'] - ingreso_spot_estres:,.0f} COP")
