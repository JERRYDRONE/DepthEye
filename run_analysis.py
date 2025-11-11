"""
Script de Análisis Personalizado - DepthEye
============================================
Análisis completo de XAUUSD usando tus datos específicos.

Este script:
1. Carga tus archivos CSV personalizados
2. Ejecuta el análisis cuantitativo completo
3. Genera visualizaciones profesionales
4. Guarda resultados en reports/
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Agregar módulos al path
sys.path.append(str(Path(__file__).parent))

from load_data import load_all_data
from xauusd_quant_analysis import (
    get_sync_pois,
    build_lot_profile,
    get_garch_forecast,
    run_monte_carlo,
    analyze_projection,
    run_complete_analysis
)
from visualization import (
    plot_lot_profile,
    plot_monte_carlo_paths,
    plot_probability_heatmap,
    create_comprehensive_dashboard
)


def main():
    """
    Función principal del análisis.
    """
    
    print("\n" + "=" * 80)
    print("DEPTHEYE - ANÁLISIS CUANTITATIVO DE XAUUSD")
    print("=" * 80)
    print()
    
    # ============================================================================
    # PASO 1: CARGAR DATOS
    # ============================================================================
    
    print("PASO 1: Cargando datos...")
    print("-" * 80)
    
    try:
        df_micro, df_sync, df_ohlc = load_all_data()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n📍 INFORMACIÓN NECESARIA:")
        print("   Por favor, proporciona la ubicación correcta de los archivos OHLC:")
        print("   - XAUUSDz_H1.csv (o similar)")
        print("   - XAUUSDz_M15.csv (opcional)")
        print()
        print("   Ejemplo: c:/Users/jerry/Dropbox/Trading/TS/XAUUSDz_H1.csv")
        return
    
    # ============================================================================
    # PASO 2: IDENTIFICAR POIs (Points of Interest)
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("PASO 2: Identificando POIs con Posiciones Sincronizadas (>500 lotes)")
    print("-" * 80)
    
    pois = get_sync_pois(
        df_sync,
        period_days=30  # Últimos 30 días
    )
    
    print(f"\n✓ Se identificaron {len(pois)} POIs")
    if len(pois) > 0:
        print("\nTop 5 POIs más recientes:")
        for i, poi in enumerate(pois[:5], 1):
            print(f"  {i}. ${poi:.2f}")
    
    # ============================================================================
    # PASO 3: PERFIL DE LOTES (MICROPOSICIONES) - SEGMENTADO MACRO/MICRO
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("PASO 3: Construyendo Perfil de Lotes con Microposiciones (Macro + Micro)")
    print("-" * 80)
    
    # Perfil MACRO (7 días) - Contexto estructural
    print("\n  → Perfil Macro (7 días - Contexto Estructural)")
    lot_profile_macro = build_lot_profile(
        df_micro,
        period_days=7,  # Últimos 7 días
        bin_size_usd=0.50  # Bins de $0.50
    )
    poc_macro = lot_profile_macro['poc']
    hlns_macro = lot_profile_macro['hlns']
    
    print(f"    ✓ POC Macro: ${poc_macro:.2f}")
    print(f"    ✓ High Liquidity Nodes (Macro): {len(hlns_macro)} niveles")
    
    # Perfil MICRO (1 día / 24 horas) - Acción inmediata
    print("\n  → Perfil Micro (24h - Acción Inmediata)")
    lot_profile_micro = build_lot_profile(
        df_micro,
        period_days=1,  # Últimas 24 horas
        bin_size_usd=0.50  # Bins de $0.50
    )
    poc_micro = lot_profile_micro['poc']
    hlns_micro = lot_profile_micro['hlns']
    llns_micro = lot_profile_micro['llns']
    
    print(f"    ✓ POC Micro: ${poc_micro:.2f}")
    print(f"    ✓ High Liquidity Nodes (Micro): {len(hlns_micro)} niveles")
    print(f"    ✓ Low Liquidity Nodes (Micro): {len(llns_micro)} niveles")
    
    # ============================================================================
    # PASO 4: MODELO GARCH DE VOLATILIDAD
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("PASO 4: Modelando Volatilidad con GARCH(1,1)")
    print("-" * 80)
    
    volatility = get_garch_forecast(
        df_ohlc,
        timeframe_hours=1.0  # H1 bars
    )
    
    if volatility is None:
        print("\n❌ ERROR: No se pudo calcular el forecast GARCH")
        return
    
    current_price = df_ohlc['close'].iloc[-1]
    
    print(f"\n✓ Forecast GARCH completado")
    print(f"   Volatilidad diaria: {volatility:.2f}%")
    print(f"   Precio actual: ${current_price:.2f}")
    
    # ============================================================================
    # PASO 5: SIMULACIÓN MONTE CARLO
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("PASO 5: Ejecutando Simulación Monte Carlo")
    print("-" * 80)
    
    # Calcular drift histórico (anualizado)
    df_ohlc['returns'] = df_ohlc['close'].pct_change()
    historical_drift = df_ohlc['returns'].mean() * 252 * 100  # Anualizado en % (252 días trading)
    
    mc_simulations = run_monte_carlo(
        start_price=current_price,
        daily_volatility=volatility,
        drift=0.0,  # Usar 0 para asumir martingala (neutral)
        num_simulations=10000,
        num_steps=24,  # 24 horas
        time_horizon=1.0  # 1 día
    )
    
    if mc_simulations.size == 0:
        print("\n❌ ERROR: No se pudo ejecutar Monte Carlo")
        return
    
    # Calcular estadísticas
    final_prices = mc_simulations[-1]
    expected_price = np.mean(final_prices)
    median_price = np.median(final_prices)
    
    print(f"\n✓ 10,000 simulaciones completadas")
    print(f"   Precio final esperado: ${expected_price:.2f}")
    print(f"   Precio final mediana: ${median_price:.2f}")
    
    # ============================================================================
    # PASO 6: ANÁLISIS DE PROBABILIDADES (TODOS LOS NIVELES: MACRO + MICRO + POIs)
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("PASO 6: Analizando Probabilidades de Objetivos")
    print("-" * 80)
    
    # Crear diccionario combinado con TODOS los niveles
    # MACRO: para contexto estructural (7d)
    # MICRO: para acción inmediata (24h) - alineado con Monte Carlo
    sr_levels_combined = {
        'poc': poc_micro,  # POC Micro es el primario para trading
        'poc_macro': poc_macro,  # POC Macro para referencia estructural
        'hlns': hlns_micro[:10],  # Top 10 HLNs Micro (accionables)
        'hlns_macro': hlns_macro[:5],  # Top 5 HLNs Macro (contexto)
        'llns': llns_micro
    }
    
    prob_analysis = analyze_projection(
        mc_matrix=mc_simulations,
        sr_levels_dict=sr_levels_combined,
        sync_pois_list=pois,
        start_price=current_price,
        tolerance=0.5
    )
    
    print(f"\n✓ Análisis completado")
    print(f"   Precio actual: ${current_price:.2f}")
    
    if not prob_analysis.empty:
        print(f"\n📊 Total de niveles analizados: {len(prob_analysis)}")
        print("\n🎯 Top 5 Objetivos con Mayor Probabilidad de Toque:")
        # Mostrar las primeras 5 filas (ya vienen ordenadas)
        for idx, row in prob_analysis.head().iterrows():
            print(f"   {row['Nivel']} - {row['Prob_Toque_%']} ({row['Tipo']}) - {row['Prioridad']} - {row['Dirección']}")
    
    # ============================================================================
    # PASO 7: GENERAR VISUALIZACIONES
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("PASO 7: Generando Visualizaciones")
    print("-" * 80)
    
    # Crear carpeta de reportes con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(f"reports/analysis_{timestamp}")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Guardando reportes en: {report_dir}")
    
    # Perfil de lotes MACRO
    print("\n1️⃣ Generando perfil de lotes Macro (7d)...")
    plot_lot_profile(
        lot_profile_dict=lot_profile_macro,
        current_price=current_price,
        save_path=str(report_dir / "perfil_lotes_macro_7d.png")
    )
    print("   ✓ perfil_lotes_macro_7d.png")
    
    # Perfil de lotes MICRO
    print("\n2️⃣ Generando perfil de lotes Micro (24h)...")
    plot_lot_profile(
        lot_profile_dict=lot_profile_micro,
        current_price=current_price,
        save_path=str(report_dir / "perfil_lotes_micro_24h.png")
    )
    print("   ✓ perfil_lotes_micro_24h.png")
    
    # Simulaciones Monte Carlo
    print("\n3️⃣ Generando simulaciones Monte Carlo...")
    # Extraer lista de niveles S/R (MICRO + POIs para alineación temporal)
    sr_levels_list = [poc_micro] + hlns_micro[:10] + pois[:5]
    plot_monte_carlo_paths(
        mc_matrix=mc_simulations,
        start_price=current_price,
        sr_levels=sr_levels_list,
        num_paths_to_plot=100,
        save_path=str(report_dir / "monte_carlo.png")
    )
    print("   ✓ monte_carlo.png")
    
    # Heatmap de probabilidades
    if not prob_analysis.empty:
        print("\n4️⃣ Generando heatmap de probabilidades...")
        plot_probability_heatmap(
            df_analysis=prob_analysis,
            save_path=str(report_dir / "heatmap_probabilidades.png")
        )
        print("   ✓ heatmap_probabilidades.png")
    
    # ============================================================================
    # PASO 8: GUARDAR RESULTADOS
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("PASO 8: Guardando Resultados")
    print("-" * 80)
    
    # Guardar resumen en CSV
    summary_data = {
        'timestamp': [datetime.now()],
        'precio_actual': [current_price],
        'precio_esperado': [expected_price],
        'precio_mediana': [median_price],
        'volatilidad_diaria': [volatility],
        'n_pois': [len(pois)],
        'n_simulaciones': [10000],
        'poc_macro_7d': [poc_macro],
        'poc_micro_24h': [poc_micro],
        'n_hlns_macro': [len(hlns_macro)],
        'n_hlns_micro': [len(hlns_micro)],
        'n_llns_micro': [len(llns_micro)]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(report_dir / "resumen.csv", index=False)
    print("\n✓ resumen.csv guardado")
    
    # Guardar POIs
    if len(pois) > 0:
        pois_df = pd.DataFrame({'price': pois})
        pois_df.to_csv(report_dir / "pois_identificados.csv", index=False)
        print("✓ pois_identificados.csv guardado")
    
    # Guardar objetivos con probabilidades
    if not prob_analysis.empty:
        prob_analysis.to_csv(report_dir / "objetivos_probabilidades.csv", index=False)
        print("✓ objetivos_probabilidades.csv guardado")
    
    # ============================================================================
    # REPORTE FINAL
    # ============================================================================
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    print()
    print(f"📊 RESUMEN EJECUTIVO:")
    print(f"   Precio Actual:           ${current_price:.2f}")
    print(f"   Precio Esperado (24h):   ${expected_price:.2f}  ({((expected_price/current_price-1)*100):+.2f}%)")
    print(f"   Precio Mediana (24h):    ${median_price:.2f}")
    print(f"   Volatilidad Diaria:      {volatility:.2f}%")
    print()
    print(f"   📍 POC Macro (7d):        ${poc_macro:.2f}  [Contexto Estructural]")
    print(f"   🎯 POC Micro (24h):       ${poc_micro:.2f}  [Acción Inmediata]")
    print()
    print(f"   POIs identificados:      {len(pois)}")
    print(f"   HLNs Macro (7d):         {len(hlns_macro)}")
    print(f"   HLNs Micro (24h):        {len(hlns_micro)}")
    print(f"   LLNs Micro (24h):        {len(llns_micro)}")
    print()
    print(f"🎯 Top 5 Objetivos con Mayor Probabilidad:")
    if not prob_analysis.empty:
        for idx, row in prob_analysis.head().iterrows():
            direction = "⬆️" if "RESISTENCIA" in row['Dirección'] else "⬇️"
            print(f"   {direction} {row['Nivel']} - {row['Prob_Toque_%']} ({row['Tipo']})")
    print()
    print(f"📁 Reportes guardados en: {report_dir}")
    print()
    print("=" * 80)


if __name__ == "__main__":
    """
    Ejecutar el análisis completo.
    """
    
    try:
        main()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Análisis interrumpido por el usuario")
        
    except Exception as e:
        print(f"\n\n❌ ERROR INESPERADO: {str(e)}")
        import traceback
        traceback.print_exc()
        print()
        print("Por favor, revisa el error y contacta si necesitas ayuda.")
