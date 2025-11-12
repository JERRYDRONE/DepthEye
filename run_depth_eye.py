"""
DepthEye - Sistema de Análisis de Profundidad de Mercado en Tiempo Real
Arquitectura: Subscriber ZMQ con Threading + Caché Híbrida (Rápido/Lento)
"""

import zmq
import json
import threading
import time
import pandas as pd
import numpy as np
import os
from datetime import datetime, timezone
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
MT5_CANDLE_COUNT = 100  # Para análisis de volatilidad

# --- PERSISTENCIA OHLC ---
# Apuntar a la carpeta 'data' local del proyecto
OHLC_CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'XAUUSDz_M1.csv')
OHLC_START_DATE = datetime(2025, 4, 7, tzinfo=timezone.utc)  # Fecha de inicio mínima (Abril 2025)

# --- CICLO LENTO (Caché) ---
CICLO_LENTO_SEGUNDOS = 900  # 15 minutos (900 seg)

# --- VARIABLES GLOBALES (Buffer Compartido) ---
realtime_lot_buffer = []
buffer_lock = threading.Lock()


def initialize_mt5():
    """Inicializa la conexión con el terminal MetaTrader 5."""
    print("\n[Setup] Conectando a MetaTrader 5...")
    if not mt5.initialize():
        print(f"❌ MT5 initialize() falló, error code = {mt5.last_error()}")
        return False

    account_info = mt5.account_info()
    if account_info is None:
        print("❌ MT5: No se pudo obtener información de la cuenta. ¿Login correcto?")
        mt5.shutdown()
        return False

    print(f"✓ MT5 Conectado: {account_info.login} en {account_info.server}")
    return True


def get_latest_candles(symbol, timeframe, count):
    """Obtiene las últimas 'count' velas de MT5."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            print(f"⚠️  MT5: No se pudieron obtener datos de {symbol}. Error: {mt5.last_error()}")
            return pd.DataFrame()

        df_rates = pd.DataFrame(rates)
        df_rates['time'] = pd.to_datetime(df_rates['time'], unit='s', utc=True)
        return df_rates

    except Exception as e:
        print(f"❌ Error en get_latest_candles: {e}")
        return pd.DataFrame()


def synchronize_historical_ohlc(symbol, timeframe, csv_path, start_date):
    """Asegura que el CSV histórico de OHLC esté completo hasta el momento actual."""
    print("\n[Setup] Sincronizando base de datos OHLC (M1)...")

    df_local = pd.DataFrame()
    last_saved_time = None

    if not os.path.exists(csv_path):
        print(f"  No se encontró CSV local. Creando archivo vacío en: {csv_path}")
        try:
            # Crear directorio si no existe
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            # Crear el archivo con cabecera estilo MT5 (tab-separated)
            mt5_header = ['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<VOL>', '<SPREAD>']
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write('\t'.join(mt5_header) + '\n')
            df_local = pd.DataFrame()
            last_saved_time = None
        except Exception as e:
            print(f"❌ Error crítico creando archivo CSV: {e}")
            return
    else:
        print("  Archivo CSV encontrado.")
        try:
            df_local = pd.read_csv(
                csv_path,
                sep='\t',
                header=0, # El archivo SÍ tiene cabecera en la fila 0
                usecols=['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<VOL>', '<SPREAD>']
            )
            # Renombrar columnas para consistencia
            df_local.rename(columns={
                '<DATE>': 'date', '<TIME>': 'time_str', 
                '<OPEN>': 'open', '<HIGH>': 'high', 
                '<LOW>': 'low', '<CLOSE>': 'close',
                '<TICKVOL>': 'tick_volume', '<VOL>': 'real_volume',
                '<SPREAD>': 'spread'
            }, inplace=True)
            df_local['time'] = pd.to_datetime(df_local['date'] + ' ' + df_local['time_str'])
            # Asumir que el datetime 'ingenuo' del CSV ya está en UTC
            df_local['time'] = df_local['time'].dt.tz_localize('UTC')
            
            if not df_local.empty:
                last_saved_time = df_local['time'].max()
                print(f"  Última vela guardada localmente: {last_saved_time}")
            else:
                print("  Archivo CSV encontrado, pero está vacío.")
        except Exception as e:
            print(f"⚠️  Error leyendo CSV local: {e}. Se re-descargará todo.")
            last_saved_time = None

    if last_saved_time is None:
        start_request_time = start_date
    else:
        start_request_time = last_saved_time + pd.Timedelta(minutes=1)

    end_request_time = datetime.now(timezone.utc)

    if start_request_time >= end_request_time:
        print("✓ Base de datos OHLC ya está al día.")
        return

    print(f"  Descargando datos M1 faltantes desde {start_request_time} hasta AHORA...")

    missing_candles_rates = mt5.copy_rates_range(symbol, timeframe, start_request_time, end_request_time)

    if missing_candles_rates is None or len(missing_candles_rates) == 0:
        print("  No se encontraron nuevas velas para rellenar.")
        return

    df_missing = pd.DataFrame(missing_candles_rates)
    df_missing['time'] = pd.to_datetime(df_missing['time'], unit='s', utc=True)

    if last_saved_time is not None:
        df_missing = df_missing[df_missing['time'] > last_saved_time]

    if df_missing.empty:
        print("  (Datos recibidos de MT5 eran duplicados, ya están guardados).")
        print("✓ Base de datos OHLC ya está al día.")
        return

    print(f"  ... {len(df_missing)} nuevas velas M1 descargadas.")

    # Preparar datos para formato MT5
    df_missing['<DATE>'] = df_missing['time'].dt.strftime('%Y.%m.%d')
    df_missing['<TIME>'] = df_missing['time'].dt.strftime('%H:%M:%S')
    # Renombrar columnas para coincidir
    df_missing.rename(columns={
        'open': '<OPEN>', 'high': '<HIGH>', 
        'low': '<LOW>', 'close': '<CLOSE>',
        'tick_volume': '<TICKVOL>', 'real_volume': '<VOL>',
        'spread': '<SPREAD>'
    }, inplace=True)

    # Seleccionar y ordenar columnas
    mt5_columns = ['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<VOL>', '<SPREAD>']
    df_to_save = df_missing[mt5_columns]

    df_to_save.to_csv(
        csv_path,
        mode='a',
        header=False, # El archivo ya tiene cabecera
        index=False,
        sep='\t'      # Usar tabuladores
    )
    print(f"✓ Base de datos OHLC sincronizada y guardada en: {csv_path}")


# ============================================================================
# FUNCIONES DE ANÁLISIS CUANTITATIVO
# ============================================================================

def load_position_data(filepath: str) -> pd.DataFrame:
    """Carga datos de posiciones (micro o sync) con formato: timestamp,price,lots"""
    print(f"📂 Cargando: {filepath}")
    
    try:
        df = pd.read_csv(filepath, header=None, names=['timestamp', 'price', 'lots'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['lots'] = pd.to_numeric(df['lots'], errors='coerce')
        df = df.dropna()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ Cargados {len(df):,} registros")
        return df
        
    except Exception as e:
        print(f"❌ Error cargando {filepath}: {e}")
        return pd.DataFrame(columns=['timestamp', 'price', 'lots'])


def get_sync_pois(df_sync: pd.DataFrame, period_days: int) -> List[float]:
    """Extrae los niveles S/R "Absolutos" o Puntos de Interés (POI) de alta prioridad"""
    try:
        if not pd.api.types.is_datetime64_any_dtype(df_sync['timestamp']):
            df_sync['timestamp'] = pd.to_datetime(df_sync['timestamp'])
        
        cutoff_date = df_sync['timestamp'].max() - pd.Timedelta(days=period_days)
        df_filtered = df_sync[df_sync['timestamp'] >= cutoff_date].copy()
        
        if df_filtered.empty:
            print(f"⚠ WARNING: No hay datos de sincronización en los últimos {period_days} días.")
            return []
        
        df_filtered = df_filtered.sort_values('lots', ascending=False)
        pois = df_filtered['price'].tolist()
        
        print(f"✓ Extraídos {len(pois)} POIs de eventos sincronizados (>{period_days}d)")
        return pois
        
    except Exception as e:
        print(f"✗ ERROR en get_sync_pois: {str(e)}")
        return []


def build_lot_profile(df_micro: pd.DataFrame, period_days: int, bin_size_usd: float = 0.25) -> Dict[str, any]:
    """Crea un Perfil de Lotes para encontrar S/R basados en la distribución de la liquidez."""
    try:
        if not pd.api.types.is_datetime64_any_dtype(df_micro['timestamp']):
            df_micro['timestamp'] = pd.to_datetime(df_micro['timestamp'])
        
        cutoff_date = df_micro['timestamp'].max() - pd.Timedelta(days=period_days)
        df_filtered = df_micro[df_micro['timestamp'] >= cutoff_date].copy()
        
        if df_filtered.empty:
            return {'poc': None, 'hlns': [], 'llns': []}
        
        min_price = df_filtered['price'].min()
        max_price = df_filtered['price'].max()
        
        bins = np.arange(min_price, max_price + bin_size_usd, bin_size_usd)
        
        df_filtered['price_bin'] = pd.cut(df_filtered['price'], bins=bins, labels=bins[:-1], include_lowest=True)
        
        lot_profile = df_filtered.groupby('price_bin')['lots'].sum().sort_index()
        lot_profile.index = lot_profile.index.astype(float)
        
        mean_lots = lot_profile.mean()
        std_lots = lot_profile.std()
        
        poc_price = lot_profile.idxmax()
        
        hln_threshold = mean_lots + std_lots
        hlns = lot_profile[lot_profile > hln_threshold].index.tolist()
        hlns = [h for h in hlns if h != poc_price]
        
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
    """Calcula la volatilidad diaria pronosticada usando la desviación estándar de los retornos M1."""
    if ohlc_df_m1.empty or len(ohlc_df_m1) < 30:
        return 0.01

    retornos_m1 = np.log(ohlc_df_m1['close'] / ohlc_df_m1['close'].shift(1)).dropna()

    if retornos_m1.empty:
        return 0.01

    vol_periodo_m1 = retornos_m1.std()
    volatilidad_diaria = vol_periodo_m1 * np.sqrt(1440)

    return volatilidad_diaria * 100


def run_monte_carlo(start_price: float, daily_volatility: float, drift: float = 0.0,
                    num_simulations: int = 10000, num_steps: int = 24, time_horizon: float = 1.0) -> np.ndarray:
    """Simula trayectorias de precios futuras usando Movimiento Browniano Geométrico (GBM)."""
    try:
        dt = time_horizon / num_steps
        sigma = daily_volatility / 100.0
        mu = drift / 100.0
        
        Z = np.random.standard_normal((num_steps, num_simulations))
        
        price_paths = np.zeros((num_steps + 1, num_simulations))
        price_paths[0] = start_price
        
        for t in range(1, num_steps + 1):
            drift_component = (mu - 0.5 * sigma**2) * dt
            diffusion_component = sigma * np.sqrt(dt) * Z[t-1]
            price_paths[t] = price_paths[t-1] * np.exp(drift_component + diffusion_component)
        
        price_paths = price_paths[1:]
        
        return price_paths
        
    except Exception as e:
        print(f"✗ ERROR en run_monte_carlo: {str(e)}")
        return np.array([])


def analyze_projection(mc_matrix: np.ndarray, sr_levels_dict: Dict[str, any],
                       sync_pois_list: List[float], start_price: float, tolerance: float = 0.5) -> pd.DataFrame:
    """Calcula la probabilidad de que el precio interactúe con los S/R cuantitativos."""
    try:
        if mc_matrix.size == 0:
            return pd.DataFrame()
        
        num_simulations = mc_matrix.shape[1]
        sr_levels = []
        
        if sr_levels_dict.get('poc'):
            sr_levels.append({'price': sr_levels_dict['poc'], 'type': 'POC_Micro', 'priority': 'ALTA'})
        
        if sr_levels_dict.get('poc_macro'):
            sr_levels.append({'price': sr_levels_dict['poc_macro'], 'type': 'POC_Macro', 'priority': 'ALTA'})
        
        for hln in sr_levels_dict.get('hlns', []):
            sr_levels.append({'price': hln, 'type': 'HLN_Micro', 'priority': 'MEDIA'})
        
        for hln in sr_levels_dict.get('hlns_macro', []):
            sr_levels.append({'price': hln, 'type': 'HLN_Macro', 'priority': 'MEDIA'})
        
        for poi in sync_pois_list[:5]:
            sr_levels.append({'price': poi, 'type': 'POI', 'priority': 'CRÍTICA'})
        
        if not sr_levels:
            return pd.DataFrame()
        
        results = []
        
        for level in sr_levels:
            price_level = level['price']
            hit_count = 0
            
            for sim_idx in range(num_simulations):
                path = mc_matrix[:, sim_idx]
                max_price = np.max(path)
                min_price = np.min(path)
                
                if price_level > start_price:
                    if max_price >= (price_level - tolerance):
                        hit_count += 1
                else:
                    if min_price <= (price_level + tolerance):
                        hit_count += 1
            
            hit_probability = (hit_count / num_simulations) * 100
            direction = "RESISTENCIA" if price_level > start_price else "SOPORTE"
            distance = abs(price_level - start_price)
            
            results.append({
                'Nivel': f"${price_level:.2f}",
                'Tipo': level['type'],
                'Dirección': direction,
                'Distancia': f"${distance:.2f}",
                'Prob_Tocar_%': f"{hit_probability:.2f}%",
                'Hit_Prob_Raw': hit_probability
            })
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('Hit_Prob_Raw', ascending=False)
        df_results = df_results.drop('Hit_Prob_Raw', axis=1)
        
        return df_results
        
    except Exception as e:
        print(f"✗ ERROR en analyze_projection: {str(e)}")
        return pd.DataFrame()


def analyze_absorption(nuevos_lotes: List[Dict], ohlc_df: pd.DataFrame) -> str:
    """
    Correlaciona los lotes en tiempo real con las velas M1 para detectar absorción.

    Args:
        nuevos_lotes: Lista de dicts de lotes del buffer ZMQ
        ohlc_df: DataFrame de las últimas 100 velas M1 de MT5

    Returns:
        String ("ABSORCIÓN SOPORTE", "DISTRIBUCIÓN RESISTENCIA", "NEUTRAL")
    """
    if not nuevos_lotes or ohlc_df.empty:
        return "NEUTRAL"

    # Convertir velas a un dict para búsqueda rápida por timestamp
    ohlc_df['time_key'] = ohlc_df['time'].dt.floor('min')
    velas_dict = ohlc_df.set_index('time_key').to_dict('index')

    total_lotes_soporte = 0
    total_lotes_resistencia = 0

    for lote in nuevos_lotes:
        try:
            # Buscar la vela M1 correspondiente
            lote_time = pd.to_datetime(lote['timestamp']).tz_localize('UTC').floor('min')
            vela = velas_dict.get(lote_time)

            if vela is None:
                continue # Lote no coincide con nuestras velas M1

            lote_price = lote['price']

            # Definir "cuerpo" y "rango" de la vela
            rango_vela = vela['high'] - vela['low']
            if rango_vela == 0: continue # Vela Doji, ignorar

            # Clasificar el lote
            # 1. ¿El lote golpeó la zona baja (soporte)?
            if (lote_price - vela['low']) / rango_vela < 0.33: # Golpeó el tercio inferior
                # ¿La vela cerró fuerte (absorción)?
                if (vela['close'] - vela['low']) / rango_vela > 0.66: # Cerró en el tercio superior
                    total_lotes_soporte += lote['lots']

            # 2. ¿El lote golpeó la zona alta (resistencia)?
            elif (vela['high'] - lote_price) / rango_vela < 0.33: # Golpeó el tercio superior
                # ¿La vela cerró débil (distribución)?
                if (vela['high'] - vela['close']) / rango_vela > 0.66: # Cerró en el tercio inferior
                    total_lotes_resistencia += lote['lots']

        except Exception:
            continue # Ignorar errores de lote

    # Decisión final del ciclo
    if total_lotes_soporte > total_lotes_resistencia and total_lotes_soporte > 50: # (Umbral de 50 lotes)
        return f"ABSORCIÓN SOPORTE ({total_lotes_soporte} lots)"
    elif total_lotes_resistencia > total_lotes_soporte and total_lotes_resistencia > 50:
        return f"DISTRIBUCIÓN RESISTENCIA ({total_lotes_resistencia} lots)"

    return "NEUTRAL"


def append_candle_to_csv(vela_df: pd.DataFrame, csv_path: str):
    """
    Anexa una nueva vela M1 al archivo CSV histórico en el formato MT5.

    Args:
        vela_df: DataFrame que contiene la *única* vela nueva
        csv_path: Ruta al archivo CSV histórico
    """
    try:
        # Preparar datos para formato MT5
        df_to_save = vela_df.copy()
        df_to_save['<DATE>'] = df_to_save['time'].dt.strftime('%Y.%m.%d')
        df_to_save['<TIME>'] = df_to_save['time'].dt.strftime('%H:%M:%S')

        # Renombrar columnas para coincidir (asegurándonos de que existan)
        df_to_save.rename(columns={
            'open': '<OPEN>', 'high': '<HIGH>', 
            'low': '<LOW>', 'close': '<CLOSE>',
            'tick_volume': '<TICKVOL>', 'real_volume': '<VOL>',
            'spread': '<SPREAD>'
        }, inplace=True)

        # Asegurar que todas las columnas MT5 existan
        mt5_columns = ['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<VOL>', '<SPREAD>']
        for col in mt5_columns:
            if col not in df_to_save.columns:
                # Asignar 0 si la columna falta (ej. 'real_volume', 'spread' de MT5)
                df_to_save[col] = 0 

        # Seleccionar y ordenar columnas
        df_to_save = df_to_save[mt5_columns]

        # Anexar al CSV (modo 'a', sin cabecera, sep='\t')
        df_to_save.to_csv(
            csv_path,
            mode='a',
            header=False, 
            index=False,
            sep='\t'
        )
    except Exception as e:
        print(f"❌ Error en append_candle_to_csv (persistencia M1): {e}")


def is_price_near_level(precio_actual: float, nivel: float, tolerancia_usd: float) -> bool:
    """Verifica si el precio actual está dentro de la zona de tolerancia de un nivel."""
    if nivel is None:
        return False
    return abs(precio_actual - nivel) <= tolerancia_usd


def find_next_sr_level(precio_actual: float, direccion: str, niveles_sr: List[float]) -> Optional[float]:
    """
    Encuentra el nivel S/R más cercano en la dirección de la operación.

    Args:
        precio_actual: El precio de entrada.
        direccion: "UP" (para TPs de Compra) or "DOWN" (para TPs de Venta).
        niveles_sr: Lista de todos los niveles S/R (POCs, HLNs, POIs).

    Returns:
        El precio del siguiente S/R, or None si no se encuentra.
    """
    if not niveles_sr:
        return None

    if direccion == "UP":
        # Buscar la resistencia más cercana (el valor MÍNIMO > precio)
        objetivos_potenciales = [lvl for lvl in niveles_sr if lvl > precio_actual]
        return min(objetivos_potenciales) if objetivos_potenciales else None
    elif direccion == "DOWN":
        # Buscar el soporte más cercano (el valor MÁXIMO < precio)
        objetivos_potenciales = [lvl for lvl in niveles_sr if lvl < precio_actual]
        return max(objetivos_potenciales) if objetivos_potenciales else None
    return None


def check_strategy_triggers(precio_actual: float, 
                            perfil_micro: Dict, 
                            perfil_macro: Dict, 
                            pois_list: List[float], 
                            estado_absorcion: str, 
                            volatilidad_diaria_pct: float) -> Dict[str, any]:
    """
    Fase 1.3: Implementación del Trigger "Alpha_MeanReversion_v1"
    Genera un "Trade Ticket" completo (Señal, Entrada, SL, TP).
    """

    # --- 1. Definir Entradas y Parámetros ---
    poc_micro = perfil_micro.get('poc')
    poc_macro = perfil_macro.get('poc')

    if poc_micro is None or poc_macro is None:
        return {"signal": "HOLD (Calculando POCs)"}

    rango_diario_esperado = (volatilidad_diaria_pct / 100.0) * precio_actual
    tolerancia_usd = max(0.50, rango_diario_esperado / 20.0)

    # Combinar TODOS los niveles S/R (eliminando Nones y duplicados)
    todos_los_sr = set(
        [poc_micro, poc_macro] + 
        perfil_macro.get('hlns', []) + 
        perfil_micro.get('hlns', []) + 
        pois_list
    )
    todos_los_sr = [lvl for lvl in todos_los_sr if lvl is not None]

    # --- 2. Lógica de SEÑAL DE VENTA (SELL) ---
    premisa_venta = (precio_actual > poc_micro) and (precio_actual > poc_macro)
    resistencias = [r for r in todos_los_sr if r > precio_actual]
    zona_venta = any(is_price_near_level(precio_actual, res, tolerancia_usd) for res in resistencias)
    confirmacion_venta = (estado_absorcion.startswith("DISTRIBUCIÓN RESISTENCIA"))

    if premisa_venta and zona_venta and confirmacion_venta:
        # --- Fase 1.3 (Riesgo) ---
        stop_loss = precio_actual + (tolerancia_usd * 2.0)
        take_profit = find_next_sr_level(precio_actual, "DOWN", todos_los_sr)

        if take_profit is None:
            return {"signal": "HOLD (Venta detectada pero sin TP claro)"}

        # Opcional: Validar R:R
        # if abs(precio_actual - take_profit) < abs(precio_actual - stop_loss):
        #     return {"signal": "HOLD (Venta detectada, R:R Pobre)"}

        return {
            "signal": "SELL",
            "entry": precio_actual,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

    # --- 3. Lógica de SEÑAL de COMPRA (BUY) ---
    premisa_compra = (precio_actual < poc_micro) and (precio_actual < poc_macro)
    soportes = [s for s in todos_los_sr if s < precio_actual]
    zona_compra = any(is_price_near_level(precio_actual, sup, tolerancia_usd) for sup in soportes)
    confirmacion_compra = (estado_absorcion.startswith("ABSORCIÓN SOPORTE"))

    if premisa_compra and zona_compra and confirmacion_compra:
        # --- Fase 1.3 (Riesgo) ---
        stop_loss = precio_actual - (tolerancia_usd * 2.0)
        take_profit = find_next_sr_level(precio_actual, "UP", todos_los_sr)

        if take_profit is None:
            return {"signal": "HOLD (Compra detectada pero sin TP claro)"}

        return {
            "signal": "BUY",
            "entry": precio_actual,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

    # --- 4. Sin Señal ---
    return {"signal": "HOLD"}


def load_historical_data():
    """Carga el archivo CSV histórico al inicio para construir el perfil Macro."""
    try:
        if not os.path.exists(DIRECTORIO_CSV):
            print(f"⚠️  Archivo histórico no encontrado: {DIRECTORIO_CSV}")
            print("   Iniciando con datos históricos vacíos...")
            return pd.DataFrame(columns=['timestamp', 'price', 'lots'])
        
        df = pd.read_csv(DIRECTORIO_CSV, names=['timestamp', 'price', 'lots'], encoding='utf-8-sig')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"✓ Cargados {len(df):,} registros históricos para el perfil Macro.")
        print(f"  Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error cargando datos históricos: {e}")
        return pd.DataFrame(columns=['timestamp', 'price', 'lots'])


def run_zmq_subscriber(buffer, lock):
    """Hilo de fondo que se conecta al Publisher ZMQ y llena el buffer."""
    print("\n[ZMQ Thread] Iniciando suscriptor...")
    
    try:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect("tcp://localhost:5555")
        socket.setsockopt_string(zmq.SUBSCRIBE, "XAUUSD_MICRO")
        
        print("[ZMQ Thread] ✓ Conectado a tcp://localhost:5555")
        print("[ZMQ Thread] ✓ Suscrito al tópico: XAUUSD_MICRO")
        print("[ZMQ Thread] Esperando datos...\n")
        
        while True:
            try:
                full_message = socket.recv_string()
                topic, data_json = full_message.split(' ', 1)
                data_packet = json.loads(data_json)
                
                with lock:
                    buffer.append(data_packet)
                
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
                time.sleep(1)
                
    except zmq.ZMQError as e:
        print(f"[ZMQ Thread] ❌ Error de ZMQ: {e}")
    except Exception as e:
        print(f"[ZMQ Thread] ❌ Error crítico: {e}")


def run_analysis_loop(df_lotes_completo, df_ohlc_completo, pois_list, 
                      volatilidad_cacheada, perfil_macro_cacheado, 
                      buffer, lock):
    """Hilo principal que ejecuta el bucle de análisis de DepthEye con caché híbrida."""
    print("\n" + "="*70)
    print("INICIANDO DEPTHEYE - Sistema de Análisis de Profundidad de Mercado")
    print("="*70)
    
    print("\n[Fase 1] Caché Inicial Cargada...")
    
    # Extraer valor de POC, que podría ser None
    poc_macro_val = perfil_macro_cacheado.get('poc')
    # Usar .get('poc') es más seguro que ['poc']
    
    # Formatear con un fallback
    poc_macro_str = f"${poc_macro_val:.2f}" if poc_macro_val is not None else "(Aún no hay datos de lotes)"
    
    print(f"✓ Lotes en RAM: {len(df_lotes_completo):,} registros")
    print(f"✓ Velas M1 en RAM: {len(df_ohlc_completo):,} registros")
    print(f"✓ POIs Sincronizados: {len(pois_list)}")
    print(f"✓ Volatilidad (Caché): {volatilidad_cacheada:.4f}%")
    print(f"✓ POC Macro (Caché): {poc_macro_str}")
    
    print("\n[Fase 2] Iniciando bucle de análisis híbrido...")
    print(f"  • Ciclo Rápido: cada 10 segundos")
    print(f"  • Ciclo Lento: cada {CICLO_LENTO_SEGUNDOS//60} minutos\n")
    
    ciclo = 0
    tiempo_ultimo_ciclo_lento = time.time()
    
    while True:
        ciclo += 1
        current_time = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n{'─'*70}")
        print(f"[Ciclo Rápido #{ciclo}] {current_time} - Análisis Principal")
        print(f"{'─'*70}")
        
        # --- PASO A: OBTENER DATOS ---
        with lock:
            nuevos_lotes = list(buffer)
            buffer.clear()

        ohlc_df_reciente = get_latest_candles(MT5_SYMBOL, MT5_TIMEFRAME, MT5_CANDLE_COUNT)

        if ohlc_df_reciente.empty:
            print("⚠️  MT5: No se recibieron datos OHLC. Saltando análisis.")
            time.sleep(10)
            continue

        precio_actual = ohlc_df_reciente.iloc[-1]['close']
        vela_m1_nueva = ohlc_df_reciente.iloc[-1:]

        # --- PASO B: ACTUALIZAR DATOS EN RAM ---
        if nuevos_lotes:
            print(f"📊 {len(nuevos_lotes)} lotes nuevos recibidos. Actualizando DF en RAM...")
            df_nuevos_lotes = pd.DataFrame(nuevos_lotes)
            df_nuevos_lotes['timestamp'] = pd.to_datetime(df_nuevos_lotes['timestamp'])
            df_lotes_completo = pd.concat([df_lotes_completo, df_nuevos_lotes], ignore_index=True)
        else:
            print("⏳ Sin lotes nuevos.")

        # Actualizar DF de OHLC en RAM (evitando duplicados)
        if not df_ohlc_completo.empty and not vela_m1_nueva.empty:
            if vela_m1_nueva.iloc[0]['time'] > df_ohlc_completo.iloc[-1]['time']:
                
                # 1. Actualizar RAM (ya existe)
                df_ohlc_completo = pd.concat([df_ohlc_completo, vela_m1_nueva], ignore_index=True)
                
                # 2. Actualizar Disco (NUEVA TAREA - Persistencia)
                append_candle_to_csv(vela_m1_nueva, OHLC_CSV_PATH)

        # --- PASO C: CICLO LENTO ---
        tiempo_actual = time.time()
        if (tiempo_actual - tiempo_ultimo_ciclo_lento) > CICLO_LENTO_SEGUNDOS:
            print("\n" + "!"*70)
            print(f"⚡ [Ciclo Lento #{ciclo}] Recalculando modelos de fondo...")
            print("!"*70)

            # 1. Recalcular Volatilidad (ya existe)
            volatilidad_cacheada = get_realized_volatility(df_ohlc_completo)

            # 2. Recalcular Perfil Macro (ya existe)
            perfil_macro_cacheado = build_lot_profile(df_lotes_completo, period_days=7, bin_size_usd=0.50)

            # --- TAREA NUEVA: RECARGAR POIs Sincronizados ---
            print("   ... Recargando POIs Sincronizados (Sync)...")
            sync_csv_path = os.path.join(USER_HOME, 'Dropbox', 'Trading', 'TS', 'ingresosXAUUSD.csv')
            df_sync = load_position_data(sync_csv_path)

            # Actualizar la lista mutable 'pois_list' (pasada como argumento)
            pois_list.clear() # Limpiar la lista anterior
            pois_list.extend(get_sync_pois(df_sync, period_days=90)) # Rellenar con los nuevos POIs
            # --- FIN DE TAREA NUEVA ---

            print(f"✓ Volatilidad Diaria (Caché) actualizada: {volatilidad_cacheada:.4f}%")
            print(f"✓ POC Macro (Caché) actualizado: ${perfil_macro_cacheado['poc']:.2f}")
            print(f"✓ POIs (Caché) actualizados: {len(pois_list)} niveles encontrados")

            tiempo_ultimo_ciclo_lento = tiempo_actual

        # --- PASO D: ANÁLISIS RÁPIDO ---
        perfil_micro = build_lot_profile(df_lotes_completo, period_days=1, bin_size_usd=0.50)

        mc_matrix = run_monte_carlo(
            start_price=precio_actual,
            daily_volatility=volatilidad_cacheada,
            drift=0.0,
            num_simulations=5000,
            num_steps=24,
            time_horizon=1.0
        )

        sr_levels_combined = {
            'poc': perfil_micro['poc'],
            'poc_macro': perfil_macro_cacheado['poc'],
            'hlns': perfil_micro.get('hlns', []),
            'hlns_macro': perfil_macro_cacheado.get('hlns', []),
        }

        prob_analysis = analyze_projection(
            mc_matrix=mc_matrix,
            sr_levels_dict=sr_levels_combined,
            sync_pois_list=pois_list,
            start_price=precio_actual
        )

        # 6. Análisis de Absorción (Fase 1.1)
        # (Usamos ohlc_df_reciente y nuevos_lotes)
        estado_absorcion = analyze_absorption(nuevos_lotes, ohlc_df_reciente)

        # 7. Generador de Señales (Fase 1.2)
        signal = check_strategy_triggers(
            precio_actual=precio_actual,
            perfil_micro=perfil_micro,
            perfil_macro=perfil_macro_cacheado,
            pois_list=pois_list,
            estado_absorcion=estado_absorcion,
            volatilidad_diaria_pct=volatilidad_cacheada
        )

        # --- PASO E: REPORTE EJECUTIVO ---
        print("\n" + "="*70)
        print(f"📊 RESUMEN EJECUTIVO (Ciclo Rápido #{ciclo})")
        print("="*70)

        final_prices = mc_matrix[-1]
        expected_price = np.mean(final_prices)

        print(f"   Precio Actual:           ${precio_actual:.2f}")
        print(f"   Precio Esperado (24h):   ${expected_price:.2f}")
        print(f"   Volatilidad Diaria (M1): {volatilidad_cacheada:.4f}% (Caché)")
        print(f"   Estado de Absorción:     {estado_absorcion}  [Fase 1.1]")
        
        # --- Impresión de Señal (Fase 1.3) ---
        signal_info = signal # 'signal' es ahora un diccionario
        if signal_info["signal"] == "HOLD":
            print(f"   🔥 SEÑAL DEL SISTEMA:   HOLD  [Fase 1.3]")
        elif signal_info["signal"].startswith("HOLD ("): # Manejar mensajes de HOLD
            print(f"   🔥 SEÑAL DEL SISTEMA:   {signal_info['signal']}  [Fase 1.3]")
        else:
            # ¡Imprimir el Ticket de Operación completo!
            print(f"   🔥🔥🔥 SEÑAL GENERADA: {signal_info['signal']} 🔥🔥🔥  [Fase 1.3]")
            print(f"     ├─ Entrada: ${signal_info['entry']:.2f}")
            print(f"     ├─ Stop Loss: ${signal_info['stop_loss']:.2f}")
            print(f"     └─ Take Profit: ${signal_info['take_profit']:.2f}")
        
        print()
        
        # Extraer valores, que podrían ser None
        poc_macro_val = perfil_macro_cacheado['poc']
        poc_micro_val = perfil_micro['poc']
        
        # Formatear con un fallback
        poc_macro_str = f"${poc_macro_val:.2f}" if poc_macro_val is not None else "(Calculando...)"
        poc_micro_str = f"${poc_micro_val:.2f}" if poc_micro_val is not None else "(Calculando...)"
        
        print(f"   📍 POC Macro (7d):        {poc_macro_str}  [Contexto - Caché]")
        print(f"   🎯 POC Micro (24h):       {poc_micro_str}  [Acción - RealTime]")
        print()

        print(f"🎯 Top 5 Objetivos con Mayor Probabilidad:")
        if not prob_analysis.empty:
            for idx, row in prob_analysis.head().iterrows():
                direction = "⬆️" if "RESISTENCIA" in row['Dirección'] else "⬇️"
                print(f"   {direction} {row['Nivel']} - {row['Prob_Tocar_%']} ({row['Tipo']})")
        else:
            print("   (Calculando probabilidades... se necesitan más datos)")

        print("="*70)

        time.sleep(10)


if __name__ == "__main__":
    print("\n" + "="*70)
    print(" DEPTHEYE v2.0 - Arquitectura de Streaming en Tiempo Real + Caché Híbrida")
    print("="*70)
    print(f" Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # --- PASO 1: Iniciar MT5 ---
    if not initialize_mt5():
        print("❌ No se pudo conectar a MT5. Saliendo.")
        exit()
    
    # --- PASO 2: Sincronizar BD de Velas M1 ---
    try:
        synchronize_historical_ohlc(MT5_SYMBOL, MT5_TIMEFRAME, OHLC_CSV_PATH, OHLC_START_DATE)
    except Exception as e:
        print(f"❌ Error Crítico durante la sincronización de OHLC: {e}")
        mt5.shutdown()
        exit()
    
    # --- PASO 2.5: Cargar BD de Velas M1 COMPLETA a RAM ---
    print("\n[Setup] Cargando base de datos OHLC M1 completa a memoria...")
    try:
        df_ohlc_completo = pd.read_csv(
            OHLC_CSV_PATH,
            sep='\t',
            header=0,
            usecols=['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<VOL>', '<SPREAD>']
        )
        # Renombrar columnas
        df_ohlc_completo.rename(columns={
            '<DATE>': 'date', '<TIME>': 'time_str', 
            '<OPEN>': 'open', '<HIGH>': 'high', 
            '<LOW>': 'low', '<CLOSE>': 'close',
            '<TICKVOL>': 'tick_volume', '<VOL>': 'real_volume',
            '<SPREAD>': 'spread'
        }, inplace=True)
        df_ohlc_completo['time'] = pd.to_datetime(df_ohlc_completo['date'] + ' ' + df_ohlc_completo['time_str'])
        # Asumir que el datetime 'ingenuo' del CSV ya está en UTC
        df_ohlc_completo['time'] = df_ohlc_completo['time'].dt.tz_localize('UTC')
        print(f"✓ Cargadas {len(df_ohlc_completo):,} velas M1 históricas.")
    except Exception as e:
        print(f"❌ Error cargando CSV de OHLC: {e}. Iniciando con DF vacío.")
        df_ohlc_completo = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close'])
    
    # --- PASO 3: Cargar datos históricos de LOTES ---
    df_lotes_completo = load_historical_data()
    
    # --- PASO 4: Cargar datos históricos de SYNC ---
    print("\n[Setup] Cargando datos históricos de POIs Sincronizados...")
    sync_csv_path = os.path.join(USER_HOME, 'Dropbox', 'Trading', 'TS', 'ingresosXAUUSD.csv')
    df_sync = load_position_data(sync_csv_path)
    
    # --- PASO 5: Obtener POIs ---
    pois_list = get_sync_pois(df_sync, period_days=90)
    
    # --- PASO 5.5: Cálculo Lento Inicial (Caché) ---
    print("\n[Setup] Realizando cálculos lentos iniciales (Macro y Volatilidad)...")
    
    volatilidad_cacheada = get_realized_volatility(df_ohlc_completo)
    perfil_macro_cacheado = build_lot_profile(df_lotes_completo, period_days=7, bin_size_usd=0.50)
    
    print(f"✓ Volatilidad Diaria (Caché): {volatilidad_cacheada:.4f}%")
    if perfil_macro_cacheado['poc'] is not None:
        print(f"✓ POC Macro (Caché): ${perfil_macro_cacheado['poc']:.2f}")
    else:
        print("✓ POC Macro (Caché): (Aún no hay datos de lotes para calcular)")
    
    # --- PASO 6: Iniciar hilo ZMQ ---
    print("\n[Setup] Iniciando hilo ZMQ Subscriber...")
    zmq_thread = threading.Thread(
        target=run_zmq_subscriber,
        args=(realtime_lot_buffer, buffer_lock),
        daemon=True,
        name="ZMQ-Subscriber"
    )
    zmq_thread.start()
    
    time.sleep(2)
    
    # --- PASO 7: Iniciar Bucle de Análisis ---
    try:
        run_analysis_loop(
            df_lotes_completo,
            df_ohlc_completo,
            pois_list,
            volatilidad_cacheada,
            perfil_macro_cacheado,
            realtime_lot_buffer,
            buffer_lock
        )
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("🛑 DepthEye detenido por el usuario")
        print("="*70)
        mt5.shutdown()
        print("✓ MT5 desconectado.")
        print(f" Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
