"""
DepthEye - Sistema de Análisis de Profundidad de Mercado en Tiempo Real
Arquitectura: Subscriber ZMQ con Threading para procesamiento asíncrono
"""

import zmq
import json
import threading
import time
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Optional
import MetaTrader5 as mt5
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURACIÓN ---
USER_HOME = os.path.expanduser('~')
DIRECTORIO_CSV = os.path.join(USER_HOME, 'Dropbox', 'Trading', 'TS', 'ingresosXAUUSD40.csv')

# --- CONFIGURACIÓN MT5 ---
MT5_SYMBOL = "XAUUSDz"
MT5_TIMEFRAME = mt5.TIMEFRAME_M1  # Velas de 1 Minuto
MT5_CANDLE_COUNT = 100  # Para análisis de volatilidad GARCH

# --- PERSISTENCIA OHLC ---
OHLC_CSV_PATH = os.path.join(USER_HOME, 'Dropbox', 'Trading', 'TS', 'historico_XAUUSDz_M1.csv')
OHLC_START_DATE = datetime(2025, 4, 7)  # Fecha de inicio mínima (Abril 2025)

# --- VARIABLES GLOBALES (Buffer Compartido) ---
realtime_lot_buffer = []
buffer_lock = threading.Lock()


def initialize_mt5():
    """
    Inicializa la conexión con el terminal MetaTrader 5.
    
    Returns:
        bool: True si la conexión fue exitosa, False en caso contrario
    """
    print("\n[Setup] Conectando a MetaTrader 5...")
    if not mt5.initialize():
        print(f"❌ MT5 initialize() falló, error code = {mt5.last_error()}")
        return False

    # Opcional: Validar login y servidor
    account_info = mt5.account_info()
    if account_info is None:
        print("❌ MT5: No se pudo obtener información de la cuenta. ¿Login correcto?")
        mt5.shutdown()
        return False

    print(f"✓ MT5 Conectado: {account_info.login} en {account_info.server}")
    return True


def get_latest_candles(symbol, timeframe, count):
    """
    Obtiene las últimas 'count' velas de MT5.
    
    Args:
        symbol (str): Símbolo del instrumento (ej: "XAUUSDz")
        timeframe: Timeframe de MT5 (ej: mt5.TIMEFRAME_M1)
        count (int): Número de velas a obtener
    
    Returns:
        pd.DataFrame: DataFrame con las velas OHLC
    """
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            print(f"⚠️  MT5: No se pudieron obtener datos de {symbol}. Error: {mt5.last_error()}")
            return pd.DataFrame()  # Retorna DF vacío

        # Convertir a DataFrame legible
        df_rates = pd.DataFrame(rates)
        df_rates['time'] = pd.to_datetime(df_rates['time'], unit='s')
        return df_rates

    except Exception as e:
        print(f"❌ Error en get_latest_candles: {e}")
        return pd.DataFrame()


def synchronize_historical_ohlc(symbol, timeframe, csv_path, start_date):
    """
    Asegura que el CSV histórico de OHLC esté completo hasta el momento actual.
    Descarga todos los datos si el archivo no existe.
    Rellena (backfills) los datos faltantes si el script estuvo apagado.
    
    Args:
        symbol (str): Símbolo del instrumento (ej: "XAUUSDz")
        timeframe: Timeframe de MT5 (ej: mt5.TIMEFRAME_M1)
        csv_path (str): Ruta al archivo CSV de persistencia
        start_date (datetime): Fecha de inicio mínima para descarga histórica
    """
    print("\n[Setup] Sincronizando base de datos OHLC (M1)...")

    # Convertir 'time' a datetime en la carga
    df_local = pd.DataFrame()
    last_saved_time = None

    if os.path.exists(csv_path):
        try:
            df_local = pd.read_csv(csv_path)
            if not df_local.empty:
                df_local['time'] = pd.to_datetime(df_local['time'])
                last_saved_time = df_local['time'].max()
                print(f"  Última vela guardada localmente: {last_saved_time}")
            else:
                print("  Archivo CSV encontrado, pero está vacío.")
        except Exception as e:
            print(f"⚠️  Error leyendo CSV local: {e}. Se re-descargará todo.")
            last_saved_time = None
    else:
        print("  No se encontró CSV local. Se descargará todo el historial.")

    # Determinar la fecha de inicio para la descarga
    if last_saved_time is None:
        # Si no hay datos, descargar desde la fecha de inicio
        start_request_time = start_date
    else:
        # Si hay datos, descargar desde la última vela + 1 minuto
        start_request_time = last_saved_time + pd.Timedelta(minutes=1)

    # Determinar la fecha de fin (ahora mismo, en UTC)
    end_request_time = datetime.utcnow()

    if start_request_time >= end_request_time:
        print("✓ Base de datos OHLC ya está al día.")
        return

    print(f"  Descargando datos M1 faltantes desde {start_request_time} hasta AHORA...")

    # Solicitar datos a MT5
    missing_candles_rates = mt5.copy_rates_range(
        symbol,
        timeframe,
        start_request_time,
        end_request_time
    )

    if missing_candles_rates is None or len(missing_candles_rates) == 0:
        print("  No se encontraron nuevas velas para rellenar.")
        return

    # Procesar y guardar los datos faltantes
    df_missing = pd.DataFrame(missing_candles_rates)
    df_missing['time'] = pd.to_datetime(df_missing['time'], unit='s')

    # Filtrar para evitar duplicados exactos en el tiempo (por si acaso)
    if last_saved_time is not None:
        df_missing = df_missing[df_missing['time'] > last_saved_time]

    if df_missing.empty:
        print("  (Datos recibidos de MT5 eran duplicados, ya están guardados).")
        print("✓ Base de datos OHLC ya está al día.")
        return

    # Anexar al CSV
    print(f"  ... {len(df_missing)} nuevas velas M1 descargadas.")

    # 'header=False' si el archivo ya existe y no está vacío
    write_header = not os.path.exists(csv_path) or df_local.empty

    df_missing.to_csv(
        csv_path,
        mode='a',
        header=write_header,
        index=False
    )
    print(f"✓ Base de datos OHLC sincronizada y guardada en: {csv_path}")


# ============================================================================
# FUNCIONES DE ANÁLISIS CUANTITATIVO (del repositorio)
# ============================================================================

def load_position_data(filepath: str) -> pd.DataFrame:
    """
    Carga datos de posiciones (micro o sync) con formato: timestamp,price,lots
    
    Args:
        filepath: Ruta al archivo CSV
        
    Returns:
        DataFrame con columnas ['timestamp', 'price', 'lots']
    """
    print(f"📂 Cargando: {filepath}")
    
    try:
        # Leer CSV sin header
        df = pd.read_csv(
            filepath,
            header=None,
            names=['timestamp', 'price', 'lots']
        )
        
        # Convertir timestamp a datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convertir a float/numeric
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['lots'] = pd.to_numeric(df['lots'], errors='coerce')
        
        # Eliminar NaNs
        df = df.dropna()
        
        # Ordenar por fecha
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ Cargados {len(df):,} registros")
        return df
        
    except Exception as e:
        print(f"❌ Error cargando {filepath}: {e}")
        return pd.DataFrame(columns=['timestamp', 'price', 'lots'])


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
        
        return {
            'poc': float(poc_price),
            'hlns': [float(h) for h in hlns],
            'llns': [float(l) for l in llns],
            'profile': lot_profile
        }
        
    except Exception as e:
        print(f"✗ ERROR en build_lot_profile: {str(e)}")
        return {'poc': None, 'hlns': [], 'llns': []}


def get_realized_volatility(ohlc_df_m1: pd.DataFrame) -> float:
    """
    Calcula la volatilidad diaria pronosticada usando la desviación estándar
    de los retornos M1 (Volatilidad Realizada).
    
    Args:
        ohlc_df_m1: DataFrame con velas M1 y columna 'close'
        
    Returns:
        Volatilidad diaria en porcentaje
    """
    if ohlc_df_m1.empty or len(ohlc_df_m1) < 30:  # Necesita suficientes datos
        return 0.01  # Un valor default bajo

    # Calcular retornos logarítmicos M1
    retornos_m1 = np.log(ohlc_df_m1['close'] / ohlc_df_m1['close'].shift(1)).dropna()

    if retornos_m1.empty:
        return 0.01

    # Calcular la volatilidad del período M1
    vol_periodo_m1 = retornos_m1.std()

    # Escalar a volatilidad diaria
    # (1440 minutos en un día de 24h)
    volatilidad_diaria = vol_periodo_m1 * np.sqrt(1440)

    # Convertir a porcentaje para el reporte
    return volatilidad_diaria * 100


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
        daily_volatility: Volatilidad diaria (sigma), en porcentaje
        drift: Retorno medio esperado (mu), default 0.0 para neutralidad
        num_simulations: Número de trayectorias a simular
        num_steps: Número de pasos temporales
        time_horizon: Horizonte temporal en días
        
    Returns:
        Matriz NumPy (num_steps, num_simulations) con todas las trayectorias
    """
    try:
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
        
        return price_paths
        
    except Exception as e:
        print(f"✗ ERROR en run_monte_carlo: {str(e)}")
        return np.array([])


def analyze_projection(mc_matrix: np.ndarray,
                       sr_levels_dict: Dict[str, any],
                       sync_pois_list: List[float],
                       start_price: float,
                       tolerance: float = 0.5) -> pd.DataFrame:
    """
    Calcula la probabilidad de que el precio interactúe con los S/R cuantitativos.
    
    Args:
        mc_matrix: Matriz de simulaciones Monte Carlo (num_steps, num_simulations)
        sr_levels_dict: Diccionario con POC, HLNs, LLNs
        sync_pois_list: Lista de POIs
        start_price: Precio inicial
        tolerance: Tolerancia en USD para considerar "toque" (default: $0.5)
        
    Returns:
        DataFrame con análisis de probabilidades para cada nivel S/R
    """
    try:
        if mc_matrix.size == 0:
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
        for hln in sr_levels_dict.get('hlns', []):
            sr_levels.append({
                'price': hln,
                'type': 'HLN_Micro',
                'priority': 'MEDIA'
            })
        
        # Agregar HLNs Macro (contexto)
        for hln in sr_levels_dict.get('hlns_macro', []):
            sr_levels.append({
                'price': hln,
                'type': 'HLN_Macro',
                'priority': 'MEDIA'
            })
        
        # Agregar POIs sincronizados (Top 5 por importancia)
        for poi in sync_pois_list[:5]:
            sr_levels.append({
                'price': poi,
                'type': 'POI',
                'priority': 'CRÍTICA'
            })
        
        if not sr_levels:
            return pd.DataFrame()
        
        # Calcular probabilidades para cada nivel
        results = []
        
        for level in sr_levels:
            price_level = level['price']
            
            # Calcular Probabilidad de Toque (Hit Probability)
            hit_count = 0
            
            for sim_idx in range(num_simulations):
                path = mc_matrix[:, sim_idx]
                max_price = np.max(path)
                min_price = np.min(path)
                
                # Determinar si hay toque (con tolerancia)
                if price_level > start_price:  # Nivel de resistencia
                    if max_price >= (price_level - tolerance):
                        hit_count += 1
                else:  # Nivel de soporte
                    if min_price <= (price_level + tolerance):
                        hit_count += 1
            
            # Calcular probabilidades
            hit_probability = (hit_count / num_simulations) * 100
            
            # Determinar dirección del nivel respecto al precio actual
            direction = "RESISTENCIA" if price_level > start_price else "SOPORTE"
            distance = abs(price_level - start_price)
            
            results.append({
                'Nivel': f"${price_level:.2f}",
                'Tipo': level['type'],
                'Dirección': direction,
                'Distancia': f"${distance:.2f}",
                'Prob_Tocar_%': f"{hit_probability:.2f}%",
                'Hit_Prob_Raw': hit_probability  # Para ordenar
            })
        
        # Crear DataFrame y ordenar por probabilidad de toque
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('Hit_Prob_Raw', ascending=False)
        df_results = df_results.drop('Hit_Prob_Raw', axis=1)
        
        return df_results
        
    except Exception as e:
        print(f"✗ ERROR en analyze_projection: {str(e)}")
        return pd.DataFrame()


def load_historical_data():
    """
    Carga el archivo CSV histórico al inicio para construir el perfil Macro.
    
    Returns:
        pd.DataFrame: DataFrame con los datos históricos
    """
    try:
        if not os.path.exists(DIRECTORIO_CSV):
            print(f"⚠️  Archivo histórico no encontrado: {DIRECTORIO_CSV}")
            print("   Iniciando con datos históricos vacíos...")
            return pd.DataFrame(columns=['timestamp', 'price', 'lots'])
        
        # Cargar CSV histórico
        df = pd.read_csv(
            DIRECTORIO_CSV,
            names=['timestamp', 'price', 'lots'],
            encoding='utf-8-sig'
        )
        
        # Convertir timestamp a datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"✓ Cargados {len(df):,} registros históricos para el perfil Macro.")
        print(f"  Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error cargando datos históricos: {e}")
        return pd.DataFrame(columns=['timestamp', 'price', 'lots'])


def run_zmq_subscriber(buffer, lock):
    """
    Hilo de fondo que se conecta al Publisher ZMQ y llena el buffer.
    
    Args:
        buffer (list): Buffer compartido para almacenar los lotes recibidos
        lock (threading.Lock): Lock para sincronización del buffer
    """
    print("\n[ZMQ Thread] Iniciando suscriptor...")
    
    try:
        # Inicializar contexto y socket ZMQ
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        
        # Conectarse al Publisher
        socket.connect("tcp://localhost:5555")
        
        # Suscribirse al tópico XAUUSD_MICRO
        socket.setsockopt_string(zmq.SUBSCRIBE, "XAUUSD_MICRO")
        
        print("[ZMQ Thread] ✓ Conectado a tcp://localhost:5555")
        print("[ZMQ Thread] ✓ Suscrito al tópico: XAUUSD_MICRO")
        print("[ZMQ Thread] Esperando datos...\n")
        
        # Bucle de recepción continua
        while True:
            try:
                # Recibir mensaje completo (tópico + JSON)
                full_message = socket.recv_string()
                
                # Separar tópico del JSON
                topic, data_json = full_message.split(' ', 1)
                
                # Parsear JSON
                data_packet = json.loads(data_json)
                
                # Sección crítica: añadir al buffer con lock
                with lock:
                    buffer.append(data_packet)
                
                # Log del dato recibido
                timestamp_str = data_packet.get('timestamp', 'N/A')
                price = data_packet.get('price', 0)
                lots = data_packet.get('lots', 0)
                
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"[ZMQ] {current_time} | Lote recibido: {lots} lots @ {price} | Buffer: {len(buffer)} items")
                
            except json.JSONDecodeError as e:
                print(f"[ZMQ] ⚠️  Error parseando JSON: {e}")
            except ValueError as e:
                print(f"[ZMQ] ⚠️  Error separando mensaje: {e}")
            except Exception as e:
                print(f"[ZMQ] ⚠️  Error inesperado: {e}")
                time.sleep(1)  # Evitar bucle infinito en caso de error
                
    except zmq.ZMQError as e:
        print(f"[ZMQ Thread] ❌ Error de ZMQ: {e}")
    except Exception as e:
        print(f"[ZMQ Thread] ❌ Error crítico: {e}")


def run_analysis_loop(historical_df, pois_list, buffer, lock):
    """
    Hilo principal que ejecuta el bucle de análisis de DepthEye.
    
    Args:
        historical_df (pd.DataFrame): Datos históricos de lotes para perfil Macro
        pois_list (List[float]): Lista de POIs sincronizados
        buffer (list): Buffer compartido con los lotes en tiempo real
        lock (threading.Lock): Lock para sincronización del buffer
    """
    print("\n" + "="*70)
    print("INICIANDO DEPTHEYE - Sistema de Análisis de Profundidad de Mercado")
    print("="*70)
    
    # Fase 1: Construir perfil Macro (datos históricos)
    print("\n[Fase 1] Construyendo Perfil Macro (7 días)...")
    if len(historical_df) > 0:
        print(f"✓ Perfil Macro (7d) construido a partir de {len(historical_df):,} registros históricos.")
        print(f"  - Precio medio: {historical_df['price'].mean():.2f}")
        print(f"  - Volumen total: {historical_df['lots'].sum():,} lots")
    else:
        print("⚠️  Sin datos históricos. Perfil Macro vacío.")
    
    print(f"\n[Fase 1.5] POIs Sincronizados cargados: {len(pois_list)}")
    
    print("\n[Fase 2] Iniciando bucle de análisis en tiempo real...")
    print("  (Procesando buffer cada 10 segundos)\n")
    
    ciclo = 0
    
    # Este DataFrame acumulará todos los lotes (históricos + tiempo real)
    df_lotes_completo = historical_df.copy()
    
    # Fase 2: Bucle de análisis continuo
    while True:
        ciclo += 1
        current_time = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n{'─'*70}")
        print(f"[Ciclo #{ciclo}] {current_time} - Análisis Principal")
        print(f"{'─'*70}")
        
        # --- PASO A: OBTENER DATOS (LOTES ZMQ Y PRECIO MT5) ---
        with lock:
            nuevos_lotes = list(buffer)
            buffer.clear()

        ohlc_df = get_latest_candles(MT5_SYMBOL, MT5_TIMEFRAME, MT5_CANDLE_COUNT)

        if ohlc_df.empty:
            print("⚠️  MT5: No se recibieron datos OHLC. Saltando análisis.")
            time.sleep(10)
            continue

        precio_actual = ohlc_df.iloc[-1]['close']

        # --- PASO B: ACTUALIZAR DATOS DE LOTES ---
        if nuevos_lotes:
            print(f"📊 {len(nuevos_lotes)} lotes nuevos recibidos. Actualizando perfiles...")
            # Convertir buffer (lista de dicts) a DataFrame
            df_nuevos_lotes = pd.DataFrame(nuevos_lotes)
            df_nuevos_lotes['timestamp'] = pd.to_datetime(df_nuevos_lotes['timestamp'])

            # Añadir al DataFrame completo (para persistencia en memoria)
            df_lotes_completo = pd.concat([df_lotes_completo, df_nuevos_lotes], ignore_index=True)
        else:
            print("⏳ Sin lotes nuevos.")

        # --- PASO C: EJECUTAR ANÁLISIS REAL ---

        # 1. Perfil Macro (7d) - Usa todos los lotes
        perfil_macro = build_lot_profile(df_lotes_completo, period_days=7, bin_size_usd=0.50)

        # 2. Perfil Micro (1d) - Usa todos los lotes
        perfil_micro = build_lot_profile(df_lotes_completo, period_days=1, bin_size_usd=0.50)

        # 3. Volatilidad (M1 Realizada)
        volatilidad_diaria = get_realized_volatility(ohlc_df)

        # 4. Monte Carlo (10k simulaciones, 24 pasos = 24 horas)
        mc_matrix = run_monte_carlo(
            start_price=precio_actual,
            daily_volatility=volatilidad_diaria,
            drift=0.0,
            num_simulations=10000,
            num_steps=24,
            time_horizon=1.0
        )

        # 5. Análisis de Probabilidades
        sr_levels_combined = {
            'poc': perfil_micro['poc'],
            'poc_macro': perfil_macro['poc'],
            'hlns': perfil_micro.get('hlns', []),
            'hlns_macro': perfil_macro.get('hlns', []),
            'llns': perfil_micro.get('llns', [])
        }

        prob_analysis = analyze_projection(
            mc_matrix=mc_matrix,
            sr_levels_dict=sr_levels_combined,
            sync_pois_list=pois_list,
            start_price=precio_actual
        )

        # --- PASO D: MOSTRAR REPORTE EJECUTIVO ---
        print("\n" + "="*70)
        print(f"� RESUMEN EJECUTIVO (Ciclo #{ciclo})")
        print("="*70)

        final_prices = mc_matrix[-1]
        expected_price = np.mean(final_prices)

        print(f"   Precio Actual:           ${precio_actual:.2f}")
        print(f"   Precio Esperado (24h):   ${expected_price:.2f}")
        print(f"   Volatilidad Diaria (M1): {volatilidad_diaria:.4f}%")
        print()
        print(f"   📍 POC Macro (7d):        ${perfil_macro['poc']:.2f}  [Contexto]")
        print(f"   🎯 POC Micro (24h):       ${perfil_micro['poc']:.2f}  [Acción]")
        print()

        print(f"🎯 Top 5 Objetivos con Mayor Probabilidad:")
        if not prob_analysis.empty:
            for idx, row in prob_analysis.head().iterrows():
                direction = "⬆️" if "RESISTENCIA" in row['Dirección'] else "⬇️"
                print(f"   {direction} {row['Nivel']} - {row['Prob_Tocar_%']} ({row['Tipo']})")
        else:
            print("   (Calculando probabilidades... se necesitan más datos)")

        print("="*70)

        time.sleep(10)  # Aumentar el ciclo a 10 segundos


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" DEPTHEYE v2.0 - Arquitectura de Streaming en Tiempo Real")
    print("="*70)
    print(f" Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # --- PASO 1: Iniciar MT5 (Obligatorio) ---
    if not initialize_mt5():
        print("❌ No se pudo conectar a MT5. Saliendo.")
        exit()
    
    # --- PASO 2: Sincronizar/Rellenar BD de Velas M1 ---
    try:
        synchronize_historical_ohlc(
            MT5_SYMBOL,
            MT5_TIMEFRAME,
            OHLC_CSV_PATH,
            OHLC_START_DATE
        )
    except Exception as e:
        print(f"❌ Error Crítico durante la sincronización de OHLC: {e}")
        mt5.shutdown()
        exit()
    
    # --- PASO 3: Cargar datos históricos de LOTES (CSV) ---
    df_macro_lotes = load_historical_data()
    
    # --- PASO 4: Cargar datos históricos de SYNC (CSV) ---
    print("\n[Setup] Cargando datos históricos de POIs Sincronizados...")
    sync_csv_path = os.path.join(USER_HOME, 'Dropbox', 'Trading', 'TS', 'ingresosXAUUSD.csv')
    df_sync = load_position_data(sync_csv_path)
    
    # --- PASO 5: Obtener POIs (Una sola vez) ---
    pois_list = get_sync_pois(df_sync, period_days=90)  # 90 días de contexto
    
    # --- PASO 6: Iniciar hilo ZMQ (Lotes en tiempo real) ---
    print("\n[Setup] Iniciando hilo ZMQ Subscriber...")
    zmq_thread = threading.Thread(
        target=run_zmq_subscriber,
        args=(realtime_lot_buffer, buffer_lock),
        daemon=True,
        name="ZMQ-Subscriber"
    )
    zmq_thread.start()
    
    # Pequeña pausa para que el subscriber se conecte
    time.sleep(2)
    
    # --- PASO 7: Iniciar Bucle de Análisis ---
    try:
        run_analysis_loop(df_macro_lotes, pois_list, realtime_lot_buffer, buffer_lock)
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("🛑 DepthEye detenido por el usuario")
        print("="*70)
        mt5.shutdown()
        print("✓ MT5 desconectado.")
        print(f" Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
