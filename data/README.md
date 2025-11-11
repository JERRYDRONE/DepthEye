# Proyecto — Datos y conversión de timestamps a UTC

Este repositorio contiene datos CSV de XAUUSD y un script para convertir los timestamps de los archivos a UTC.

Contenido y formato de los datos

1) `ingresosXAUUSD40.csv` — Microposiciones (>25 lotes)

 - Formato: sin cabecera
 - Columnas: `timestamp,price,lots`
 - Ejemplo de línea: `2025-11-09 18:36:18,4011.94,26`

2) `ingresosXAUUSD.csv` — Posiciones sincronizadas (>500 lotes)

 - Formato: sin cabecera
 - Columnas: `timestamp,price,lots`
 - Ejemplo: `2025-04-07 05:51:24,3037.82,294`

3) `XAUUSDz_H1.csv` — Velas OHLC H1 exportadas desde MetaTrader

 - Tab-separated, con cabecera en la primera línea
 - Columnas típicas: `<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>` (ej. `2025.04.07 06:00:00 ...`)

Requisitos (para ejecutar localmente)

 - Python 3.8+ (recomendado)
 - Virtualenv (opcional pero recomendado)
 - Dependencias: ver `requirements.txt` (actualmente: pandas)

Instalación rápida en un equipo personal

1. Clona el repo y entra en la carpeta `data`.
2. Crea y activa un entorno virtual:

   python -m venv .venv
   # PowerShell
   .\.venv\Scripts\Activate.ps1

3. Instala dependencias:

   pip install -r requirements.txt

Uso: convertir timestamps a UTC

El script principal es `adjust_timestamps.py`. Soporta dos modos:

- Zona IANA: `--source-tz <IANA>` (aplica DST según la fecha).
- Offset fijo: `--fixed-offset <hours>` (ej. `-6` para CDMX si consideras que no tiene DST).

Comandos de ejemplo (ejecutar desde la carpeta `data`)

- Dry-run (no escribe):

  python .\adjust_timestamps.py --input .\ingresosXAUUSD.csv --timestamp-col 0 --dry-run --fixed-offset -6

- Convertir sólo `ingresosXAUUSD40.csv` y crear un nuevo archivo con sufijo `_utc`:

  python .\adjust_timestamps.py --input .\ingresosXAUUSD40.csv --timestamp-col 0 --fixed-offset -6

- Convertir sólo `ingresosXAUUSD.csv` y crear un nuevo archivo con sufijo `_utc`:

  python .\adjust_timestamps.py --input .\ingresosXAUUSD.csv --timestamp-col 0 --fixed-offset -6

- Convertir in-place (sobrescribe el CSV):

  python .\adjust_timestamps.py --input .\ingresosXAUUSD.csv --timestamp-col 0 --fixed-offset -6 --inplace

- Convertir ambos archivos con PowerShell (usa el Python del virtualenv si está activado):

  .\convert_ingresos_inplace.ps1

- Archivo con fecha y hora en columnas separadas (ej. `XAUUSDz_H1.csv`):

  python .\adjust_timestamps.py --input .\XAUUSDz_H1.csv --date-col "<DATE>" --time-col "<TIME>" --fixed-offset -6 --inplace

Notas importantes

- Aplicar `--fixed-offset -6` equivale a sumar 6 horas al timestamp (UTC = local + 6h). Tanto la hora como la fecha se ajustan; si la suma cruza medianoche, la fecha cambiará.
- Reaplicar la conversión a un archivo ya convertido suma otras 6 horas (efecto acumulativo). Para evitarlo:
  - Usa `--dry-run` antes de sobrescribir.
  - Evita reconvertir archivos que ya terminen en `_utc`.
  - Si quieres, puedo añadir detección automática (flag `--skip-if-utc`) para prevenir reconversiones accidentales.
- Si tus timestamps pertenecen a una zona con DST real, usa `--source-tz <IANA>` en vez de `--fixed-offset`.

Archivos y scripts relevantes

- `adjust_timestamps.py` — conversor principal (usa `--fixed-offset` o `--source-tz`).
- `convert_ingresos_inplace.ps1` — PowerShell para convertir ambos `ingresosXAUUSD.csv` e `ingresosXAUUSD40.csv` in-place con `--fixed-offset -6`.
- `requirements.txt` — dependencias (pandas).

Soporte y siguientes pasos

Si quieres que implemente detección automática para evitar doble-conversiones (`--skip-if-utc`) o que la salida use formato ISO con zona (`YYYY-MM-DDTHH:MM:SS+00:00`), dime cuál prefieres y lo implemento.
