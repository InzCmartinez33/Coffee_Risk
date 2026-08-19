# 1. EJECUCIÓN DE LAS PROYECCIONES
# ---------------------------------------------------------------------
# Asegúrate de asignar el retorno a la variable `dict_proy`
dict_proy = calcular_todas_las_proyecciones(params, df_mercado)

# Extraemos las variables que usa la aplicación
trayectorias_carga = dict_proy["trayectorias_carga"]
trayectorias_cafe = dict_proy["trayectorias_cafe"]
trayectorias_trm = dict_proy["trayectorias_trm"]
precios_futuros = dict_proy["precios_futuros"]
var_95 = dict_proy["var_95"]
precio_promedio_sim = dict_proy["precio_promedio_sim"]
trm_promedio_sim = dict_proy["trm_promedio_sim"]

# Para los gráficos individuales tomamos el último día simulado
cafe_futuro = trayectorias_cafe[-1, :]
trm_futura = trayectorias_trm[-1, :]


# 2. PROYECCIONES INDIVIDUALES (CAFÉ Y TRM)
# ---------------------------------------------------------------------
st.subheader(f"2. Proyecciones Individuales de Mercado a {dias_analisis} Días")

if len(cafe_futuro) > 0 and len(trm_futura) > 0:
    # Ahora dict_proy sí existirá en el contexto global
    trm_prom_sim = dict_proy["trm_promedio_sim"]
    cafe_prom_sim = np.mean(cafe_futuro) / 100.0

    cafe_p5 = np.percentile(cafe_futuro, 5) / 100.0
    cafe_p95 = np.percentile(cafe_futuro, 95) / 100.0

    trm_p5 = np.percentile(trm_futura, 5)
    trm_p95 = np.percentile(trm_futura, 95)

    col_cafe, col_trm = st.columns(2)

    with col_cafe:
        st.markdown("#### ☕ Proyección Café NY (USD/lb)")
        st.metric(
            "Precio Promedio Simulado", 
            f"${cafe_prom_sim:.2f} USD/lb", 
            delta=f"{(cafe_prom_sim - (precio_ny_hoy/100)):.2f} USD"
        )
        st.caption(f"📉 **Escenario Crítico (P5%):** ${cafe_p5:.2f} USD/lb")
        st.caption(f"📈 **Escenario Alcista (P95%):** ${cafe_p95:.2f} USD/lb")

        fig_c, ax_c = plt.subplots(figsize=(6, 3))
        ax_c.hist(cafe_futuro / 100.0, bins=40, color='#B45309', alpha=0.7, edgecolor='white')
        ax_c.axvline(precio_ny_hoy/100, color='#10B981', linestyle='--', label='Hoy')
        ax_c.axvline(cafe_prom_sim, color='#1E3A8A', linestyle='-', label='Promedio Proyectado')
        ax_c.set_title("Distribución Café NY", fontsize=9, fontweight='bold')
        ax_c.set_xlabel("USD / lb", fontsize=8)
        ax_c.grid(True, alpha=0.3)
        ax_c.legend(fontsize=7)
        st.pyplot(fig_c)

    with col_trm:
        st.markdown("#### 💵 Proyección Dólar TRM (COP/USD)")
        st.metric(
            "TRM Promedio Simulada", 
            f"${trm_prom_sim:,.2f} COP", 
            delta=f"{(trm_prom_sim - trm_hoy):,.2f} COP"
        )
        st.caption(f"📉 **Escenario Crítico (P5%):** ${trm_p5:,.2f} COP")
        st.caption(f"📈 **Escenario Alcista (P95%):** ${trm_p95:,.2f} COP")

        fig_t, ax_t = plt.subplots(figsize=(6, 3))
        ax_t.hist(trm_futura, bins=40, color='#047857', alpha=0.7, edgecolor='white')
        ax_t.axvline(trm_hoy, color='#10B981', linestyle='--', label='Hoy')
        ax_t.axvline(trm_prom_sim, color='#1E3A8A', linestyle='-', label='Promedio Proyectado')
        ax_t.set_title("Distribución TRM", fontsize=9, fontweight='bold')
        ax_t.set_xlabel("COP / USD", fontsize=8)
        ax_t.grid(True, alpha=0.3)
        ax_t.legend(fontsize=7)
        st.pyplot(fig_t)
