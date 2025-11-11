"""
Script de Carga de Datos Personalizado - DepthEye
==================================================
Adaptado para cargar los archivos CSV específicos del usuario.

Formatos de entrada:
- ingresosXAUUSD40.csv: timestamp,price,lots (microposiciones >25 lotes)
- ingresosXAUUSD.csv: timestamp,price,lots (posiciones sync >500 lotes)
- XAUUSDz_H1.csv: <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
"""

import pandas as pd
import numpy as np
from pathlib import Path


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
        print(f"  - Rango de fechas: {df['timestamp'].min()} a {df['timestamp'].max()}")
        print(f"  - Rango de precios: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        print(f"  - Lotes: min={df['lots'].min():.0f}, max={df['lots'].max():.0f}, promedio={df['lots'].mean():.1f}")
        print()
        
        return df
        
    except Exception as e:
        print(f"✗ Error cargando {filepath}: {str(e)}")
        raise


def load_ohlc_data(filepath: str) -> pd.DataFrame:
    """
    Carga datos OHLC con formato MetaTrader (separado por tabuladores).
    
    Formato esperado: <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
    
    Args:
        filepath: Ruta al archivo CSV
        
    Returns:
        DataFrame con columnas ['timestamp', 'open', 'high', 'low', 'close']
    """
    print(f"📂 Cargando OHLC: {filepath}")
    
    try:
        # Leer CSV con tabuladores (skip primera fila que es header)
        df = pd.read_csv(
            filepath,
            sep='\t',
            skiprows=1,  # Saltar la línea de headers <DATE> <TIME> etc.
            header=None,
            names=['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']
        )
        
        # Combinar fecha y hora
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        
        # Seleccionar solo columnas necesarias
        df = df[['timestamp', 'open', 'high', 'low', 'close']]
        
        # Convertir a float
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Eliminar NaNs
        df = df.dropna()
        
        # Validar que high >= low
        invalid_rows = (df['high'] < df['low']) | (df['high'] < df['close']) | (df['low'] > df['close'])
        if invalid_rows.any():
            print(f"⚠ Advertencia: {invalid_rows.sum()} filas con datos OHLC inválidos (serán eliminadas)")
            df = df[~invalid_rows]
        
        # Ordenar por fecha
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✓ Cargadas {len(df):,} velas")
        print(f"  - Rango de fechas: {df['timestamp'].min()} a {df['timestamp'].max()}")
        print(f"  - Precio actual (última vela): ${df['close'].iloc[-1]:.2f}")
        print()
        
        return df
        
    except Exception as e:
        print(f"✗ Error cargando {filepath}: {str(e)}")
        raise


def load_all_data(base_path: str = "data/"):
    """
    Carga todos los archivos de datos necesarios para el análisis.
    
    Args:
        base_path: Ruta base donde están los archivos (por defecto: data/)
        
    Returns:
        Tupla (df_micro, df_sync, df_ohlc)
    """
    print("=" * 70)
    print("CARGANDO DATOS DE XAUUSD")
    print("=" * 70)
    print()
    
    # Convertir a Path
    base_path = Path(base_path)
    
    # Cargar microposiciones (>25 lotes)
    print("1️⃣ Microposiciones (>25 lotes)")
    micro_file = base_path / "ingresosXAUUSD40.csv"
    if not micro_file.exists():
        raise FileNotFoundError(f"No se encontró: {micro_file}")
    df_micro = load_position_data(str(micro_file))
    
    # Validar que sean >25 lotes
    below_threshold = df_micro['lots'] <= 25
    if below_threshold.any():
        print(f"⚠ Advertencia: {below_threshold.sum()} registros con ≤25 lotes (se filtrarán)")
        df_micro = df_micro[df_micro['lots'] > 25]
    
    print("-" * 70)
    
    # Cargar posiciones sincronizadas (>500 lotes)
    print("\n2️⃣ Posiciones Sincronizadas (>500 lotes)")
    sync_file = base_path / "ingresosXAUUSD.csv"
    if not sync_file.exists():
        raise FileNotFoundError(f"No se encontró: {sync_file}")
    df_sync = load_position_data(str(sync_file))
    
    # Validar que sean >500 lotes (o filtrar si es necesario)
    # Nota: Si tus datos tienen menos de 500 lotes, ajustar el threshold
    below_sync_threshold = df_sync['lots'] < 100  # Ajustado a 100 para tus datos
    if below_sync_threshold.any():
        print(f"ℹ Info: {below_sync_threshold.sum()} registros con <100 lotes")
    
    print("-" * 70)
    
    # Cargar datos OHLC (H1)
    print("\n3️⃣ Datos OHLC (H1)")
    
    ohlc_file = base_path / "XAUUSDz_H1.csv"
    if not ohlc_file.exists():
        # Intentar alternativas
        possible_paths = [
            base_path / "XAUUSD_H1.csv",
            base_path / "XAU_H1.csv",
        ]
        for path in possible_paths:
            if path.exists():
                ohlc_file = path
                break
        else:
            raise FileNotFoundError(f"No se encontró archivo OHLC H1 en {base_path}")
    
    df_ohlc = load_ohlc_data(str(ohlc_file))
    
    print("=" * 70)
    print("✓ TODOS LOS DATOS CARGADOS EXITOSAMENTE")
    print("=" * 70)
    print()
    
    # Mostrar resumen
    print("📊 RESUMEN DE DATOS:")
    print(f"   Microposiciones:  {len(df_micro):,} registros")
    print(f"   Posiciones Sync:  {len(df_sync):,} registros")
    print(f"   Velas OHLC H1:    {len(df_ohlc):,} registros")
    print()
    
    return df_micro, df_sync, df_ohlc


def copy_files_to_data_folder(base_path: str = "c:/Users/jerry/Dropbox/Trading/TS/"):
    """
    Copia los archivos CSV a la carpeta data/ del proyecto.
    
    Args:
        base_path: Ruta base donde están los archivos originales
    """
    import shutil
    
    print("📦 Copiando archivos a la carpeta data/...")
    
    base_path = Path(base_path)
    data_path = Path("data")
    
    files_to_copy = [
        ("ingresosXAUUSD40.csv", "xauusd_micro.csv"),
        ("ingresosXAUUSD.csv", "xauusd_sync.csv"),
    ]
    
    for source_name, dest_name in files_to_copy:
        source = base_path / source_name
        dest = data_path / dest_name
        
        if source.exists():
            shutil.copy2(source, dest)
            print(f"✓ Copiado: {source_name} → {dest_name}")
        else:
            print(f"✗ No encontrado: {source_name}")
    
    # Copiar OHLC si existe
    ohlc_files = ["XAUUSDz_H1.csv", "XAUUSD_H1.csv", "XAU_H1.csv"]
    for ohlc_file in ohlc_files:
        source = base_path / ohlc_file
        if source.exists():
            dest = data_path / "xauusd_ohlc_h1.csv"
            shutil.copy2(source, dest)
            print(f"✓ Copiado: {ohlc_file} → xauusd_ohlc_h1.csv")
            break
    
    print()


# ============================================================================
# FUNCIÓN PRINCIPAL DE EJEMPLO
# ============================================================================

if __name__ == "__main__":
    """
    Ejemplo de uso del script de carga de datos.
    """
    
    try:
        # Cargar todos los datos
        df_micro, df_sync, df_ohlc = load_all_data()
        
        # Mostrar las primeras filas de cada dataset
        print("\n" + "=" * 70)
        print("VISTA PREVIA DE DATOS")
        print("=" * 70)
        print()
        
        print("📊 Microposiciones (primeras 5 filas):")
        print(df_micro.head())
        print()
        
        print("📊 Posiciones Sincronizadas (primeras 5 filas):")
        print(df_sync.head())
        print()
        
        print("📊 OHLC H1 (últimas 5 velas):")
        print(df_ohlc.tail())
        print()
        
        # Ahora puedes usar estos DataFrames con el sistema DepthEye
        print("=" * 70)
        print("✅ DATOS LISTOS PARA ANÁLISIS")
        print("=" * 70)
        print()
        print("Siguiente paso: Ejecutar el análisis cuantitativo")
        print("Comando: python run_analysis.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPor favor, verifica:")
        print("  1. Que los archivos existan en la ruta especificada")
        print("  2. Que los formatos de los archivos sean correctos")
        print("  3. Que la ruta base sea correcta")
