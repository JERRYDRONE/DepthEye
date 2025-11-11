"""
Script de Análisis Rápido - DepthEye
=====================================
Ejecuta el análisis cuantitativo completo sin generar todas las visualizaciones.
Ideal para obtener resultados rápidos.
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
    analyze_projection
)


def main():
    """
    Función principal del análisis rápido.
    """
    
    print("\n" + "=" * 80)
    print("DEPTHEYE - ANÁLISIS RÁPIDO DE XAUUSD")
    print("=" * 80)
    print()
    
    # Cargar datos
    print("Cargando datos...")
    df_micro, df_sync, df_ohlc = load_all_data()
    
    # POIs
    print("\nIdentificando POIs...")
    pois = get_sync_pois(df_sync, period_days=30)
    print(f"✓ {len(pois)} POIs identificados")
    
    # Perfil de lotes - SEGMENTADO (Macro + Micro)
    print("\nConstruyendo perfil de lotes...")
    
    # Perfil MACRO (7 días) - Contexto estructural
    print("  → Perfil Macro (7 días - Contexto)")
    lot_profile_macro = build_lot_profile(df_micro, period_days=7, bin_size_usd=0.50)
    poc_macro = lot_profile_macro['poc']
    hlns_macro = lot_profile_macro['hlns']
    print(f"    ✓ POC Macro: ${poc_macro:.2f}")
    print(f"    ✓ {len(hlns_macro)} HLNs Macro identificados")
    
    # Perfil MICRO (1 día) - Acción inmediata
    print("  → Perfil Micro (24h - Acción)")
    lot_profile_micro = build_lot_profile(df_micro, period_days=1, bin_size_usd=0.50)
    poc_micro = lot_profile_micro['poc']
    hlns_micro = lot_profile_micro['hlns']
    print(f"    ✓ POC Micro: ${poc_micro:.2f}")
    print(f"    ✓ {len(hlns_micro)} HLNs Micro identificados")
    
    # GARCH
    print("\nCalculando volatilidad GARCH...")
    volatility = get_garch_forecast(df_ohlc, timeframe_hours=1.0)
    current_price = df_ohlc['close'].iloc[-1]
    print(f"✓ Volatilidad diaria: {volatility:.2f}%")
    print(f"✓ Precio actual: ${current_price:.2f}")
    
    # Monte Carlo
    print("\nEjecutando Monte Carlo (10,000 simulaciones)...")
    mc_simulations = run_monte_carlo(
        start_price=current_price,
        daily_volatility=volatility,
        drift=0.0,
        num_simulations=10000,
        num_steps=24,
        time_horizon=1.0
    )
    
    final_prices = mc_simulations[-1]
    expected_price = np.mean(final_prices)
    median_price = np.median(final_prices)
    percentile_5 = np.percentile(final_prices, 5)
    percentile_95 = np.percentile(final_prices, 95)
    
    print(f"✓ Precio esperado (24h): ${expected_price:.2f}")
    print(f"✓ Rango 90%: ${percentile_5:.2f} - ${percentile_95:.2f}")
    
    # Análisis de probabilidades - CON TODOS LOS NIVELES (Macro + Micro + POIs)
    print("\nAnalizando probabilidades...")
    
    # Crear diccionario combinado con TODOS los niveles
    # MACRO: para contexto estructural
    # MICRO: para acción inmediata
    sr_levels_combined = {
        'poc': poc_micro,  # POC Micro es el primario para trading
        'poc_macro': poc_macro,  # POC Macro para referencia
        'hlns': hlns_micro[:10],  # Top 10 HLNs Micro (accionables)
        'hlns_macro': hlns_macro[:5],  # Top 5 HLNs Macro (contexto)
        'llns': lot_profile_micro.get('llns', [])
    }
    
    prob_analysis = analyze_projection(
        mc_matrix=mc_simulations,
        sr_levels_dict=sr_levels_combined,
        sync_pois_list=pois,
        start_price=current_price,
        tolerance=0.5
    )
    
    # Crear carpeta de reportes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(f"reports/analysis_{timestamp}")
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar resultados
    print(f"\nGuardando resultados en {report_dir}...")
    
    # Guardar resumen en CSV
    summary_data = {
        'timestamp': [datetime.now()],
        'precio_actual': [current_price],
        'precio_esperado_24h': [expected_price],
        'precio_mediana_24h': [median_price],
        'percentil_5': [percentile_5],
        'percentil_95': [percentile_95],
        'volatilidad_diaria_pct': [volatility],
        'poc_macro_7d': [poc_macro],
        'poc_micro_24h': [poc_micro],
        'n_pois': [len(pois)],
        'n_hlns_macro': [len(hlns_macro)],
        'n_hlns_micro': [len(hlns_micro)],
        'n_llns_micro': [len(lot_profile_micro.get('llns', []))]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(report_dir / "resumen.csv", index=False)
    
    # POIs
    if len(pois) > 0:
        pois_df = pd.DataFrame({'price': pois})
        pois_df['distance_from_current'] = pois_df['price'] - current_price
        pois_df['distance_pct'] = (pois_df['distance_from_current'] / current_price) * 100
        pois_df.to_csv(report_dir / "pois.csv", index=False)
    
    # Probabilidades
    if not prob_analysis.empty:
        prob_analysis.to_csv(report_dir / "probabilidades.csv", index=False)
    
    # HLNs Macro y Micro
    if len(hlns_macro) > 0:
        hlns_macro_df = pd.DataFrame({'price': hlns_macro})
        hlns_macro_df['distance_from_current'] = hlns_macro_df['price'] - current_price
        hlns_macro_df['distance_pct'] = (hlns_macro_df['distance_from_current'] / current_price) * 100
        hlns_macro_df.to_csv(report_dir / "hlns_macro_7d.csv", index=False)
    
    if len(hlns_micro) > 0:
        hlns_micro_df = pd.DataFrame({'price': hlns_micro})
        hlns_micro_df['distance_from_current'] = hlns_micro_df['price'] - current_price
        hlns_micro_df['distance_pct'] = (hlns_micro_df['distance_from_current'] / current_price) * 100
        hlns_micro_df.to_csv(report_dir / "hlns_micro_24h.csv", index=False)
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS COMPLETADO")
    print("=" * 80)
    print()
    print("📊 RESULTADOS:")
    print(f"   Precio Actual:           ${current_price:.2f}")
    print(f"   Precio Esperado (24h):   ${expected_price:.2f}  ({((expected_price/current_price-1)*100):+.2f}%)")
    print(f"   Rango 90%:               ${percentile_5:.2f} - ${percentile_95:.2f}")
    print(f"   Volatilidad Diaria:      {volatility:.2f}%")
    print()
    print(f"   📍 POC Macro (7d):        ${poc_macro:.2f}  [Contexto Estructural]")
    print(f"   🎯 POC Micro (24h):       ${poc_micro:.2f}  [Acción Inmediata]")
    print()
    print(f"   POIs identificados:      {len(pois)}")
    print(f"   HLNs Macro (7d):         {len(hlns_macro)}")
    print(f"   HLNs Micro (24h):        {len(hlns_micro)}")
    print()
    print(f"🎯 Top 5 Objetivos con Mayor Probabilidad:")
    for idx, row in prob_analysis.head().iterrows():
        direction = "⬆️" if "RESISTENCIA" in row['Dirección'] else "⬇️"
        print(f"   {direction} {row['Nivel']} - {row['Prob_Toque_%']} ({row['Tipo']})")
    print()
    print(f"📁 Reportes guardados en: {report_dir}")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Análisis interrumpido")
    except Exception as e:
        print(f"\n\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
