"""
XAUUSD Quantitative Analysis Script
====================================
Ingeniero de Software Cuantitativo - Análisis de Flujo de Órdenes y Proyección de Volatilidad

Autor: Quant Engineer
Fecha: 10 de noviembre de 2025

Módulos:
1. Identificación Cuantitativa de S/R (Lot Profile & POIs)
2. Modelo de Volatilidad (GARCH)
3. Simulación de Proyección (Monte Carlo)
4. Integración y Análisis Final
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MÓDULO 1: IDENTIFICACIÓN CUANTITATIVA DE S/R (LOT PROFILE & POIs)
# ============================================================================

def get_sync_pois(df_sync: pd.DataFrame, period_days: int) -> List[float]:
    """
    Extrae los niveles S/R "Absolutos" o Puntos de Interés (POI) de alta prioridad
    basados en posiciones sincronizadas (>500 lotes).
    
    Args:
        df_sync: DataFrame con columnas ['timestamp', 'price', 'lots']
        period_days: Número de días hacia atrás para filtrar
        
    Returns:
        Lista de precios que representan POIs críticos
    """
    try:
        # Asegurar que timestamp es datetime
        if not pd.api.types.is_datetime64_any_dtype(df_sync['timestamp']):
            df_sync['timestamp'] = pd.to_datetime(df_sync['timestamp'])
        
        # Calcular el cutoff date
        cutoff_date = df_sync['timestamp'].max() - pd.Timedelta(days=period_days)
        
        # Filtrar por período
        df_filtered = df_sync[df_sync['timestamp'] >= cutoff_date].copy()
        
        if df_filtered.empty:
            print(f"⚠ WARNING: No hay datos de sincronización en los últimos {period_days} días.")
            return []
        
        # Extraer los niveles de precio (ordenados por volumen de lotes)
        df_filtered = df_filtered.sort_values('lots', ascending=False)
        pois = df_filtered['price'].tolist()
        
        print(f"✓ Extraídos {len(pois)} POIs de eventos sincronizados (>{period_days}d)")
        return pois
        
    except Exception as e:
        print(f"✗ ERROR en get_sync_pois: {str(e)}")
        return []


def build_lot_profile(df_micro: pd.DataFrame, 
                      period_days: int, 
                      bin_size_usd: float = 0.25) -> Dict[str, any]:
    """
    Crea un "Perfil de Lotes" (análogo a un Perfil de Volumen) para encontrar
    S/R basados en la distribución de la liquidez.
    
    Args:
        df_micro: DataFrame con columnas ['timestamp', 'price', 'lots']
        period_days: Número de días hacia atrás para analizar
        bin_size_usd: Tamaño del bin de precio en USD (default: $0.25)
        
    Returns:
        Diccionario con {'poc': float, 'hlns': [floats], 'llns': [floats]}
    """
    try:
        # Asegurar que timestamp es datetime
        if not pd.api.types.is_datetime64_any_dtype(df_micro['timestamp']):
            df_micro['timestamp'] = pd.to_datetime(df_micro['timestamp'])
        
        # Calcular el cutoff date
        cutoff_date = df_micro['timestamp'].max() - pd.Timedelta(days=period_days)
        
        # Filtrar por período
        df_filtered = df_micro[df_micro['timestamp'] >= cutoff_date].copy()
        
        if df_filtered.empty:
            print(f"✗ ERROR: No hay datos de microposiciones en los últimos {period_days} días.")
            return {'poc': None, 'hlns': [], 'llns': []}
        
        # Encontrar rango de precios
        min_price = df_filtered['price'].min()
        max_price = df_filtered['price'].max()
        
        # Crear bins de precio
        bins = np.arange(min_price, max_price + bin_size_usd, bin_size_usd)
        
        # Asignar cada precio a un bin
        df_filtered['price_bin'] = pd.cut(
            df_filtered['price'], 
            bins=bins, 
            labels=bins[:-1],
            include_lowest=True
        )
        
        # Agrupar por bin y sumar lotes
        lot_profile = df_filtered.groupby('price_bin')['lots'].sum().sort_index()
        
        # Convertir el índice a float
        lot_profile.index = lot_profile.index.astype(float)
        
        # Calcular estadísticas
        mean_lots = lot_profile.mean()
        std_lots = lot_profile.std()
        
        # Identificar POC (Point of Control) - Máximo volumen
        poc_price = lot_profile.idxmax()
        
        # Identificar HLN (High Lotage Nodes) - > 1 std por encima de la media
        hln_threshold = mean_lots + std_lots
        hlns = lot_profile[lot_profile > hln_threshold].index.tolist()
        # Remover el POC de HLNs si está presente
        hlns = [h for h in hlns if h != poc_price]
        
        # Identificar LLN (Low Lotage Nodes) - < 1 std por debajo de la media
        lln_threshold = mean_lots - std_lots
        llns = lot_profile[lot_profile < lln_threshold].index.tolist()
        
        print(f"✓ Perfil de Lotes construido ({period_days}d, bin=${bin_size_usd})")
        print(f"  - Rango de precios: ${min_price:.2f} - ${max_price:.2f}")
        print(f"  - Total de bins: {len(lot_profile)}")
        print(f"  - POC identificado: ${poc_price:.2f}")
        print(f"  - HLNs identificados: {len(hlns)}")
        print(f"  - LLNs identificados: {len(llns)}")
        
        return {
            'poc': float(poc_price),
            'hlns': [float(h) for h in hlns],
            'llns': [float(l) for l in llns],
            'profile': lot_profile  # Para análisis adicional o visualización
        }
        
    except Exception as e:
        print(f"✗ ERROR en build_lot_profile: {str(e)}")
        return {'poc': None, 'hlns': [], 'llns': []}


# ============================================================================
# MÓDULO 2: MODELO DE VOLATILIDAD (GARCH)
# ============================================================================

def get_garch_forecast(df_ohlc: pd.DataFrame, 
                       timeframe_hours: float = 1.0) -> Optional[float]:
    """
    Calcula el régimen de volatilidad actual y pronostica para el siguiente período
    usando un modelo GARCH(1,1).
    
    Args:
        df_ohlc: DataFrame con columnas ['timestamp', 'open', 'high', 'low', 'close']
        timeframe_hours: Timeframe de las velas en horas (1.0 para H1, 0.25 para M15)
        
    Returns:
        Volatilidad diaria pronosticada (sigma) como porcentaje
    """
    try:
        # Verificar que hay suficientes datos
        min_observations = 200
        if len(df_ohlc) < min_observations:
            print(f"✗ ERROR: Se necesitan al menos {min_observations} observaciones. Disponibles: {len(df_ohlc)}")
            return None
        
        # Calcular retornos logarítmicos
        df_ohlc = df_ohlc.copy()
        df_ohlc['returns'] = np.log(df_ohlc['close'] / df_ohlc['close'].shift(1)) * 100
        
        # Eliminar NaNs
        returns = df_ohlc['returns'].dropna()
        
        if len(returns) < min_observations:
            print(f"✗ ERROR: Retornos insuficientes después de limpieza: {len(returns)}")
            return None
        
        print(f"✓ Calculando modelo GARCH con {len(returns)} observaciones...")
        
        # Ajustar modelo GARCH(1,1)
        # mean='Zero' asume que el retorno medio es cercano a cero (común en trading de alta frecuencia)
        model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', rescale=False)
        
        # Entrenar el modelo
        res = model.fit(disp='off', show_warning=False)
        
        # Pronosticar volatilidad para el siguiente período
        forecast = res.forecast(horizon=1)
        
        # Extraer la varianza pronosticada y convertir a desviación estándar
        variance_forecast = forecast.variance.values[-1, 0]
        volatility_period = np.sqrt(variance_forecast)
        
        # Escalar la volatilidad al timeframe diario
        # Si tenemos datos H1, hay 24 períodos por día
        periods_per_day = 24.0 / timeframe_hours
        daily_volatility = volatility_period * np.sqrt(periods_per_day)
        
        print(f"✓ Modelo GARCH ajustado exitosamente")
        print(f"  - Volatilidad del período ({timeframe_hours}h): {volatility_period:.4f}%")
        print(f"  - Volatilidad diaria pronosticada: {daily_volatility:.4f}%")
        print(f"  - AIC: {res.aic:.2f}, BIC: {res.bic:.2f}")
        
        return daily_volatility
        
    except Exception as e:
        print(f"✗ ERROR en get_garch_forecast: {str(e)}")
        return None


# ============================================================================
# MÓDULO 3: SIMULACIÓN DE PROYECCIÓN (MONTE CARLO)
# ============================================================================

def run_monte_carlo(start_price: float,
                    daily_volatility: float,
                    drift: float = 0.0,
                    num_simulations: int = 10000,
                    num_steps: int = 24,
                    time_horizon: float = 1.0) -> np.ndarray:
    """
    Simula trayectorias de precios futuras usando Movimiento Browniano Geométrico (GBM).
    
    Args:
        start_price: Precio inicial de XAUUSD
        daily_volatility: Volatilidad diaria (sigma) del Módulo 2, en porcentaje
        drift: Retorno medio esperado (mu), default 0.0 para neutralidad
        num_simulations: Número de trayectorias a simular
        num_steps: Número de pasos temporales
        time_horizon: Horizonte temporal en días
        
    Returns:
        Matriz NumPy (num_steps, num_simulations) con todas las trayectorias
    """
    try:
        print(f"✓ Ejecutando simulación Monte Carlo...")
        print(f"  - Precio inicial: ${start_price:.2f}")
        print(f"  - Volatilidad: {daily_volatility:.4f}%")
        print(f"  - Simulaciones: {num_simulations:,}")
        print(f"  - Pasos: {num_steps}")
        print(f"  - Horizonte: {time_horizon} días")
        
        # Calcular el incremento de tiempo
        dt = time_horizon / num_steps
        
        # Convertir volatilidad de porcentaje a decimal
        sigma = daily_volatility / 100.0
        mu = drift / 100.0
        
        # Generar matriz de números aleatorios normales
        Z = np.random.standard_normal((num_steps, num_simulations))
        
        # Inicializar matriz de precios
        price_paths = np.zeros((num_steps + 1, num_simulations))
        price_paths[0] = start_price
        
        # Calcular trayectorias usando GBM
        # S_t = S_{t-1} * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z_t)
        for t in range(1, num_steps + 1):
            drift_component = (mu - 0.5 * sigma**2) * dt
            diffusion_component = sigma * np.sqrt(dt) * Z[t-1]
            price_paths[t] = price_paths[t-1] * np.exp(drift_component + diffusion_component)
        
        # Eliminar el precio inicial (solo devolver proyecciones)
        price_paths = price_paths[1:]
        
        # Calcular estadísticas de la simulación
        final_prices = price_paths[-1]
        mean_final = np.mean(final_prices)
        median_final = np.median(final_prices)
        std_final = np.std(final_prices)
        
        print(f"✓ Simulación completada")
        print(f"  - Precio final promedio: ${mean_final:.2f}")
        print(f"  - Precio final mediana: ${median_final:.2f}")
        print(f"  - Desviación estándar final: ${std_final:.2f}")
        
        return price_paths
        
    except Exception as e:
        print(f"✗ ERROR en run_monte_carlo: {str(e)}")
        return np.array([])


# ============================================================================
# MÓDULO 4: INTEGRACIÓN Y ANÁLISIS FINAL
# ============================================================================

def analyze_projection(mc_matrix: np.ndarray,
                       sr_levels_dict: Dict[str, any],
                       sync_pois_list: List[float],
                       start_price: float,
                       tolerance: float = 0.5) -> pd.DataFrame:
    """
    Calcula la probabilidad de que el precio interactúe con los S/R cuantitativos.
    
    Args:
        mc_matrix: Matriz de simulaciones Monte Carlo (num_steps, num_simulations)
        sr_levels_dict: Diccionario con POC, HLNs, LLNs del Módulo 1.2
        sync_pois_list: Lista de POIs del Módulo 1.1
        start_price: Precio inicial
        tolerance: Tolerancia en USD para considerar "toque" (default: $0.5)
        
    Returns:
        DataFrame con análisis de probabilidades para cada nivel S/R
    """
    try:
        print(f"\n✓ Analizando probabilidades de interacción con S/R...")
        
        if mc_matrix.size == 0:
            print("✗ ERROR: Matriz de Monte Carlo vacía")
            return pd.DataFrame()
        
        num_simulations = mc_matrix.shape[1]
        
        # Compilar todos los niveles S/R
        sr_levels = []
        
        # Agregar POC Micro (primario para trading)
        if sr_levels_dict.get('poc'):
            sr_levels.append({
                'price': sr_levels_dict['poc'],
                'type': 'POC_Micro',
                'priority': 'ALTA'
            })
        
        # Agregar POC Macro (contexto estructural)
        if sr_levels_dict.get('poc_macro'):
            sr_levels.append({
                'price': sr_levels_dict['poc_macro'],
                'type': 'POC_Macro',
                'priority': 'ALTA'
            })
        
        # Agregar HLNs Micro (accionables)
        for i, hln in enumerate(sr_levels_dict.get('hlns', []), 1):
            sr_levels.append({
                'price': hln,
                'type': 'HLN_Micro',
                'priority': 'MEDIA'
            })
        
        # Agregar HLNs Macro (contexto)
        for i, hln in enumerate(sr_levels_dict.get('hlns_macro', []), 1):
            sr_levels.append({
                'price': hln,
                'type': 'HLN_Macro',
                'priority': 'MEDIA'
            })
        
        # Agregar LLNs
        for i, lln in enumerate(sr_levels_dict.get('llns', []), 1):
            sr_levels.append({
                'price': lln,
                'type': 'LLN',
                'priority': 'BAJA'
            })
        
        # Agregar POIs sincronizados (Top 5 por importancia)
        for i, poi in enumerate(sync_pois_list[:5], 1):
            sr_levels.append({
                'price': poi,
                'type': 'POI',
                'priority': 'CRÍTICA'
            })
        
        if not sr_levels:
            print("⚠ WARNING: No hay niveles S/R para analizar")
            return pd.DataFrame()
        
        # Calcular probabilidades para cada nivel
        results = []
        
        for level in sr_levels:
            price_level = level['price']
            
            # Calcular Probabilidad de Toque (Hit Probability)
            # ¿Cuántas simulaciones cruzaron este nivel?
            hit_count = 0
            close_above_count = 0
            close_below_count = 0
            
            for sim_idx in range(num_simulations):
                path = mc_matrix[:, sim_idx]
                max_price = np.max(path)
                min_price = np.min(path)
                final_price = path[-1]
                
                # Determinar si hay toque (con tolerancia)
                if price_level > start_price:  # Nivel de resistencia
                    if max_price >= (price_level - tolerance):
                        hit_count += 1
                else:  # Nivel de soporte
                    if min_price <= (price_level + tolerance):
                        hit_count += 1
                
                # Determinar posición final
                if final_price > price_level:
                    close_above_count += 1
                else:
                    close_below_count += 1
            
            # Calcular probabilidades
            hit_probability = (hit_count / num_simulations) * 100
            close_above_prob = (close_above_count / num_simulations) * 100
            close_below_prob = (close_below_count / num_simulations) * 100
            
            # Determinar dirección del nivel respecto al precio actual
            direction = "RESISTENCIA" if price_level > start_price else "SOPORTE"
            distance = abs(price_level - start_price)
            distance_pct = (distance / start_price) * 100
            
            results.append({
                'Nivel': f"${price_level:.2f}",
                'Tipo': level['type'],
                'Prioridad': level['priority'],
                'Dirección': direction,
                'Distancia_USD': f"${distance:.2f}",
                'Distancia_%': f"{distance_pct:.3f}%",
                'Prob_Toque_%': f"{hit_probability:.2f}%",
                'Prob_Cierre_Arriba_%': f"{close_above_prob:.2f}%",
                'Prob_Cierre_Abajo_%': f"{close_below_prob:.2f}%",
                'Hit_Prob_Raw': hit_probability  # Para ordenar
            })
        
        # Crear DataFrame y ordenar por probabilidad de toque
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('Hit_Prob_Raw', ascending=False)
        df_results = df_results.drop('Hit_Prob_Raw', axis=1)
        
        print(f"✓ Análisis completado para {len(df_results)} niveles S/R")
        
        return df_results
        
    except Exception as e:
        print(f"✗ ERROR en analyze_projection: {str(e)}")
        return pd.DataFrame()


# ============================================================================
# FUNCIÓN PRINCIPAL: ORQUESTACIÓN DEL ANÁLISIS COMPLETO
# ============================================================================

def run_complete_analysis(df_micro: pd.DataFrame,
                          df_sync: pd.DataFrame,
                          df_ohlc: pd.DataFrame,
                          lot_profile_days: int = 7,
                          sync_poi_days: int = 30,
                          bin_size: float = 0.25,
                          timeframe_hours: float = 1.0,
                          num_simulations: int = 10000,
                          num_steps: int = 24,
                          time_horizon: float = 1.0) -> Dict:
    """
    Ejecuta el análisis cuantitativo completo de XAUUSD.
    
    Args:
        df_micro: DataFrame de microposiciones
        df_sync: DataFrame de posiciones sincronizadas
        df_ohlc: DataFrame de datos OHLC
        lot_profile_days: Días para el perfil de lotes
        sync_poi_days: Días para POIs sincronizados
        bin_size: Tamaño del bin en USD
        timeframe_hours: Timeframe de las velas OHLC
        num_simulations: Número de simulaciones Monte Carlo
        num_steps: Pasos de la simulación
        time_horizon: Horizonte temporal en días
        
    Returns:
        Diccionario con todos los resultados del análisis
    """
    print("=" * 70)
    print("ANÁLISIS CUANTITATIVO XAUUSD - DEPTH EYE")
    print("=" * 70)
    print()
    
    # MÓDULO 1: Identificación de S/R
    print("[MÓDULO 1] IDENTIFICACIÓN CUANTITATIVA DE S/R")
    print("-" * 70)
    
    sync_pois = get_sync_pois(df_sync, sync_poi_days)
    lot_profile = build_lot_profile(df_micro, lot_profile_days, bin_size)
    
    print()
    
    # MÓDULO 2: Modelo GARCH
    print("[MÓDULO 2] MODELO DE VOLATILIDAD GARCH(1,1)")
    print("-" * 70)
    
    daily_volatility = get_garch_forecast(df_ohlc, timeframe_hours)
    
    if daily_volatility is None:
        print("✗ ANÁLISIS ABORTADO: No se pudo calcular la volatilidad GARCH")
        return {}
    
    print()
    
    # MÓDULO 3: Simulación Monte Carlo
    print("[MÓDULO 3] SIMULACIÓN MONTE CARLO (GBM)")
    print("-" * 70)
    
    start_price = df_ohlc['close'].iloc[-1]
    mc_matrix = run_monte_carlo(
        start_price=start_price,
        daily_volatility=daily_volatility,
        drift=0.0,  # Neutral para trading de corto plazo
        num_simulations=num_simulations,
        num_steps=num_steps,
        time_horizon=time_horizon
    )
    
    if mc_matrix.size == 0:
        print("✗ ANÁLISIS ABORTADO: Fallo en la simulación Monte Carlo")
        return {}
    
    print()
    
    # MÓDULO 4: Integración y Análisis
    print("[MÓDULO 4] INTEGRACIÓN Y ANÁLISIS DE PROBABILIDADES")
    print("-" * 70)
    
    df_analysis = analyze_projection(
        mc_matrix=mc_matrix,
        sr_levels_dict=lot_profile,
        sync_pois_list=sync_pois,
        start_price=start_price
    )
    
    # Generar reporte final
    print()
    print("=" * 70)
    print("REPORTE FINAL - ANÁLISIS CUANTITATIVO XAUUSD")
    print("=" * 70)
    print()
    print(f"📊 CONTEXTO DEL MERCADO")
    print(f"   Precio Actual:                    ${start_price:.2f}")
    print(f"   Volatilidad Diaria (GARCH):       {daily_volatility:.4f}%")
    print(f"   Período de Análisis (Lot Profile): {lot_profile_days} días")
    print(f"   Período de Análisis (POIs Sync):   {sync_poi_days} días")
    print()
    
    print(f"🎯 NIVELES S/R IDENTIFICADOS")
    if lot_profile['poc']:
        print(f"   POC (Point of Control):           ${lot_profile['poc']:.2f}")
    print(f"   HLNs (High Lotage Nodes):         {len(lot_profile['hlns'])} niveles")
    print(f"   LLNs (Low Lotage Nodes):          {len(lot_profile['llns'])} niveles")
    print(f"   POIs Sincronizados (>500 lotes):  {len(sync_pois)} niveles")
    print()
    
    print(f"🎲 PROYECCIÓN MONTE CARLO")
    print(f"   Número de Simulaciones:           {num_simulations:,}")
    print(f"   Pasos Temporales:                 {num_steps}")
    print(f"   Horizonte:                        {time_horizon} día(s)")
    print()
    
    if not df_analysis.empty:
        print(f"📈 PROBABILIDADES DE INTERACCIÓN CON S/R")
        print()
        print(df_analysis.to_string(index=False))
    else:
        print("⚠ No se generaron resultados de probabilidad")
    
    print()
    print("=" * 70)
    
    # Retornar todos los componentes del análisis
    return {
        'start_price': start_price,
        'daily_volatility': daily_volatility,
        'lot_profile': lot_profile,
        'sync_pois': sync_pois,
        'mc_matrix': mc_matrix,
        'probability_analysis': df_analysis
    }


# ============================================================================
# EJEMPLO DE USO Y DATOS DE PRUEBA
# ============================================================================

if __name__ == "__main__":
    """
    Sección de ejemplo con datos sintéticos para demostrar el flujo completo.
    En producción, reemplazar con datos reales de tu broker/feed.
    """
    
    print("Generando datos de prueba para demostración...")
    print()
    
    # Generar datos sintéticos de microposiciones (>25 lotes)
    np.random.seed(42)
    n_micro = 5000
    
    dates_micro = pd.date_range(end=pd.Timestamp.now(), periods=n_micro, freq='5T')
    base_price_micro = 2350.0
    
    df_micro = pd.DataFrame({
        'timestamp': dates_micro,
        'price': base_price_micro + np.random.normal(0, 5, n_micro) + np.cumsum(np.random.normal(0, 0.1, n_micro)),
        'lots': np.random.exponential(50, n_micro) + 25
    })
    
    # Generar datos sintéticos de posiciones sincronizadas (>500 lotes)
    n_sync = 50
    dates_sync = pd.date_range(end=pd.Timestamp.now(), periods=n_sync, freq='12H')
    
    df_sync = pd.DataFrame({
        'timestamp': dates_sync,
        'price': base_price_micro + np.random.normal(0, 10, n_sync),
        'lots': np.random.exponential(200, n_sync) + 500
    })
    
    # Generar datos sintéticos OHLC (H1)
    n_ohlc = 2000
    dates_ohlc = pd.date_range(end=pd.Timestamp.now(), periods=n_ohlc, freq='1H')
    
    # Simular precios con proceso GARCH-like
    returns = np.random.normal(0, 1.2, n_ohlc)
    prices = base_price_micro * np.exp(np.cumsum(returns / 100))
    
    df_ohlc = pd.DataFrame({
        'timestamp': dates_ohlc,
        'open': prices,
        'high': prices + np.abs(np.random.normal(0, 2, n_ohlc)),
        'low': prices - np.abs(np.random.normal(0, 2, n_ohlc)),
        'close': prices + np.random.normal(0, 1, n_ohlc)
    })
    
    # Ejecutar análisis completo
    results = run_complete_analysis(
        df_micro=df_micro,
        df_sync=df_sync,
        df_ohlc=df_ohlc,
        lot_profile_days=7,
        sync_poi_days=30,
        bin_size=0.50,  # $0.50 bins para XAUUSD
        timeframe_hours=1.0,  # H1 timeframe
        num_simulations=10000,
        num_steps=24,  # 24 horas
        time_horizon=1.0  # 1 día
    )
    
    # Los resultados están disponibles en el diccionario 'results'
    # Puedes acceder a componentes individuales para análisis adicional:
    # - results['start_price']
    # - results['daily_volatility']
    # - results['lot_profile']
    # - results['sync_pois']
    # - results['mc_matrix']
    # - results['probability_analysis']
