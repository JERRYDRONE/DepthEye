#!/usr/bin/env python3
"""
adjust_timestamps.py

Convierte timestamps en CSV a UTC.

Soporta:
- Zona IANA (--source-tz) -> aplica DST según la fecha.
- Offset fijo (--fixed-offset) -> aplica un desplazamiento horario constante (ej -6 para UTC-6 fijo).

El script detecta separador (',' o '\t'), soporta columna única de timestamp (índice 0-based)
o columnas separadas de fecha y hora (pasar --date-col y --time-col con nombres si el CSV tiene cabecera).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
from zoneinfo import ZoneInfo
from typing import Optional


def detect_sep(path: Path) -> str:
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        first = f.readline()
        return '\t' if '\t' in first else ','


def load_csv(path: Path, sep: str, header: Optional[int]):
    return pd.read_csv(path, sep=sep, header=header, dtype=str, keep_default_na=False)


def parse_and_convert(df: pd.DataFrame, *, timestamp_col: Optional[int], date_col: Optional[str], time_col: Optional[str], date_format: Optional[str], source_tz: Optional[str], fixed_offset: Optional[float]) -> pd.DataFrame:
    if timestamp_col is not None:
        col = df.columns[timestamp_col]
        ts = pd.to_datetime(df[col], format=date_format, errors='coerce')
    else:
        if date_col not in df.columns or time_col not in df.columns:
            raise ValueError('Columnas de fecha/hora no encontradas en el CSV')
        combined = df[date_col].str.strip() + ' ' + df[time_col].str.strip()
        ts = pd.to_datetime(combined, format=date_format, errors='coerce')

    if ts.isnull().any():
        print('Advertencia: algunas fechas no pudieron parsearse y serán NaT', file=sys.stderr)

    if fixed_offset is not None:
        # UTC = local - offset_hours
        ts_utc = ts - pd.to_timedelta(fixed_offset, unit='h')
    else:
        if not source_tz:
            raise ValueError('Debe proporcionar --source-tz o --fixed-offset')
        try:
            _ = ZoneInfo(source_tz)
        except Exception:
            raise ValueError(f"Zona inválida: {source_tz}")
        ts_loc = ts.dt.tz_localize(source_tz, ambiguous='infer', nonexistent='shift_forward')
        ts_utc = ts_loc.dt.tz_convert('UTC')

    out_fmt = date_format if date_format else '%Y-%m-%d %H:%M:%S'
    ts_out = ts_utc.dt.strftime(out_fmt)

    if timestamp_col is not None:
        df[df.columns[timestamp_col]] = ts_out
    else:
        df[date_col] = ts_out.str.slice(0, 10)
        df[time_col] = ts_out.str.slice(11)

    return df


def main(argv=None):
    p = argparse.ArgumentParser(description='Convertir timestamps en CSV a UTC')
    p.add_argument('--input', '-i', required=True, help='Archivo CSV de entrada')
    p.add_argument('--output', '-o', help='Archivo CSV de salida (si no se especifica, se añade _utc)')
    p.add_argument('--timestamp-col', '-t', type=int, help='Índice 0-based de columna timestamp (por defecto 0)')
    p.add_argument('--date-col', help='Nombre de columna fecha (si aplica)')
    p.add_argument('--time-col', help='Nombre de columna hora (si aplica)')
    p.add_argument('--date-format', help='Formato para parsear/format (ej: "%Y-%m-%d %H:%M:%S")')
    p.add_argument('--source-tz', help='Zona IANA de origen (ej: America/Mexico_City)')
    p.add_argument('--fixed-offset', type=float, help='Offset fijo en horas respecto a UTC (ej: -6)')
    p.add_argument('--inplace', action='store_true', help='Sobrescribir archivo de entrada')
    p.add_argument('--sep', help='Separador del CSV (auto detecta , o \t)')
    p.add_argument('--header', type=int, choices=[0, -1], default=None, help='0 si tiene cabecera, -1 si no')
    p.add_argument('--dry-run', action='store_true', help='No escribe, solo muestra resumen')
    args = p.parse_args(argv)

    path = Path(args.input)
    if not path.exists():
        p.error('Archivo no encontrado: ' + args.input)

    sep = args.sep if args.sep else detect_sep(path)

    header = None
    if args.header is not None:
        header = 0 if args.header == 0 else None
    else:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            first = f.readline()
            header = 0 if any(c.isalpha() for c in first) else None

    df = load_csv(path, sep=sep, header=header)

    ts_col = args.timestamp_col if args.timestamp_col is not None else 0
    fixed_offset = args.fixed_offset

    if header is None and (args.date_col or args.time_col):
        p.error('No se puede usar --date-col/--time-col en archivos sin cabecera')

    if header is None:
        df_out = parse_and_convert(df, timestamp_col=ts_col, date_col=None, time_col=None, date_format=args.date_format, source_tz=args.source_tz, fixed_offset=fixed_offset)
    else:
        if args.date_col and args.time_col:
            df_out = parse_and_convert(df, timestamp_col=None, date_col=args.date_col, time_col=args.time_col, date_format=args.date_format, source_tz=args.source_tz, fixed_offset=fixed_offset)
        else:
            df_out = parse_and_convert(df, timestamp_col=ts_col, date_col=None, time_col=None, date_format=args.date_format, source_tz=args.source_tz, fixed_offset=fixed_offset)

    out_path = Path(args.output) if args.output else path.with_name(path.stem + '_utc' + path.suffix)
    if args.inplace:
        out_path = path

    if args.dry_run:
        print(f"Dry-run: {path} -> {out_path} (sep='{sep}')\nFilas: {len(df_out)}")
        return 0

    df_out.to_csv(out_path, sep=sep, index=False, header=(header == 0))
    print('Archivo escrito:', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
