# DepthEye - Análisis Cuantitativo XAUUSD

Sistema de análisis cuantitativo para trading de XAUUSD que combina flujo de lotes, volatilidad GARCH(1,1) y simulaciones Monte Carlo.

**Versión:** 2.0 | **Estado:** ✅ Producción

---

## 🚀 Inicio Rápido

### 1. Activar entorno virtual
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Ejecutar análisis

```powershell
# Análisis rápido (15 segundos) - Solo CSVs
python -X utf8 quick_analysis.py

# Análisis completo (3 minutos) - CSVs + Gráficos PNG
python -X utf8 run_analysis.py
```

### 3. Ver resultados

Los reportes se guardan en `reports/analysis_TIMESTAMP/`:
- **resumen.csv** - Métricas principales
- **objetivos_probabilidades.csv** - Probabilidades de toque para cada nivel S/R
- **pois_identificados.csv** - Points of Interest identificados
- **perfil_lotes_macro_7d.png** - Gráfico de perfil semanal *(solo run_analysis.py)*
- **perfil_lotes_micro_24h.png** - Gráfico de perfil diario *(solo run_analysis.py)*
- **monte_carlo.png** - Simulaciones de precio *(solo run_analysis.py)*
- **heatmap_probabilidades.png** - Heatmap de probabilidades *(solo run_analysis.py)*

---

## 📊 ¿Qué hace el sistema?

### Análisis en 6 pasos:

1. **Carga de datos**
   - Microposiciones (>25 lotes): `ingresosXAUUSD40.csv`
   - Posiciones sincronizadas (>500 lotes): `ingresosXAUUSD.csv`
   - Velas OHLC H1: `XAUUSDz_H1.csv`

2. **Identificación de POIs**
   - Detecta niveles con posiciones sincronizadas masivas (>500 lotes)
   - Son zonas donde instituciones/grandes traders entraron simultáneamente

3. **Construcción de Perfiles de Lotes** (Metodología Dual)
   - **Perfil Macro (7 días):** Contexto estructural del mercado
   - **Perfil Micro (24 horas):** Niveles accionables para trading inmediato
   - Identifica POC, HLNs y LLNs en cada perfil

4. **Cálculo de Volatilidad GARCH(1,1)**
   - Modelo econométrico que pronostica volatilidad diaria
   - Captura "clustering" de volatilidad (períodos volátiles tienden a seguir volátiles)

5. **Simulación Monte Carlo (10,000 trayectorias)**
   - Proyecta 10,000 escenarios posibles del precio en 24 horas
   - Usa volatilidad GARCH como input
   - Horizonte alineado con Perfil Micro (24h)

6. **Análisis de Probabilidades**
   - Calcula probabilidad de toque para cada nivel S/R
   - Prioriza: POIs > POC_Micro > HLN_Micro
   - Genera **Top 5** con objetivos accionables

---

## 📖 Glosario de Términos

### Niveles Soporte/Resistencia

**POC (Point of Control)**
- Nivel de precio con **mayor volumen de lotes** en un período
- **POC Macro (7d):** Mayor volumen semanal → Contexto estructural
- **POC Micro (24h):** Mayor volumen diario → Accionable HOY
- *Analogía:* El "campo de batalla" donde más traders están posicionados

**HLN (High Liquidity Node)**
- Zonas de precio con **alta concentración de lotes**
- **HLN Macro:** Soportes/resistencias estructurales (7 días)
- **HLN Micro:** Niveles accionables para scalping (24 horas)
- *Uso:* El precio tiende a reaccionar en estas zonas (bounce o ruptura)

**LLN (Low Liquidity Node)**
- Zonas de precio con **baja concentración de lotes**
- El precio puede atravesarlas rápidamente sin resistencia
- *Uso:* Identificar zonas de potencial aceleración

**POI (Point of Interest)**
- Niveles con **posiciones sincronizadas masivas** (>500 lotes)
- Prioridad **CRÍTICA** en el análisis
- *Significado:* Grandes traders entraron al mismo tiempo → Nivel importante

### Modelos Estadísticos

**GARCH(1,1)**
- *Generalized AutoRegressive Conditional Heteroskedasticity*
- Modelo econométrico para pronosticar volatilidad
- Captura que la volatilidad tiende a agruparse en períodos
- **Output:** Volatilidad diaria esperada (ej: 0.98%)

**Monte Carlo**
- Técnica de simulación que genera miles de escenarios posibles
- En DepthEye: 10,000 trayectorias de precio en 24 horas
- Usa volatilidad GARCH como input
- **Output:** Probabilidades de toque para cada nivel S/R

### Métricas de Probabilidad

**Prob. Toque**
- Probabilidad (%) de que el precio **toque** ese nivel en 24h
- No significa que se rompa, solo que llegue a ese precio
- **Uso:** Niveles con >70% son objetivos de alta confianza

**Prob. Cierre Arriba/Abajo**
- Probabilidad de que el precio **cierre** por encima/debajo del nivel
- Complementa la prob. de toque para analizar dirección

**Top 5**
- Los 5 objetivos con mayor probabilidad de toque
- Priorizados: POIs cercanos > POC_Micro > HLN_Micro
- **⬆️** = RESISTENCIA (arriba del precio actual)
- **⬇️** = SOPORTE (abajo del precio actual)

---

## 🎯 Interpretación de Resultados

### Ejemplo de salida en consola:

```
📊 RESULTADOS:
   Precio Actual:           $4110.84
   Precio Esperado (24h):   $4111.19  (+0.01%)
   Volatilidad Diaria:      0.98%

   📍 POC Macro (7d):        $4080.44  [Contexto Estructural]
   🎯 POC Micro (24h):       $4080.44  [Acción Inmediata]

   POIs identificados:      17
   HLNs Macro (7d):         30
   HLNs Micro (24h):        30

🎯 Top 5 Objetivos:
   ⬆️ $4114.87 - 82.90% (POI)          ← Resistencia crítica
   ⬇️ $4080.44 - 40.43% (POC_Micro)    ← Soporte principal
   ⬇️ $4079.44 - 38.94% (HLN_Micro)    ← Soporte secundario
   ⬇️ $4078.94 - 37.38% (HLN_Micro)
   ⬇️ $4078.44 - 36.82% (HLN_Micro)
```

### Cómo usar el Top 5 para trading:

**Escenario:** Precio actual $4,110.84

1. **Watch $4,114.87** (Resistencia - 83% probabilidad)
   - Si toca → Buscar señal de venta (rechazo)
   - Si rompe → Resistencia invalidada, buscar siguiente nivel

2. **Si baja, watch $4,080.44** (Soporte - 40% probabilidad)
   - Si toca → Buscar señal de compra (bounce)
   - Si rompe → Siguiente soporte en $4,079.44

3. **Convergencia de POCs (ambos en $4,080.44)**
   - Nivel ULTRA fuerte (consenso temporal semanal y diario)
   - Alta probabilidad de reacción significativa

### ¿Por qué dos perfiles (Macro + Micro)?

**Problema que resuelve:** Evitar "ruido metodológico" al mezclar horizontes temporales.

- **Solo Macro (7d):** POC puede estar muy lejos para trading de 24h → Top 5 lleno de niveles irrelevantes
- **Solo Micro (24h):** Pierdes visión del contexto estructural del mercado

**Solución:** Dos perfiles, dos propósitos:
- **Macro (7d):** "¿Dónde está el campo de batalla esta semana?" → Contexto
- **Micro (24h):** "¿Qué niveles puedo tradear hoy?" → Acción

El Top 5 se genera con Micro + POIs (alineación temporal con Monte Carlo 24h), pero ves ambos POCs para contexto completo.

---

## ⚙️ Configuración y Personalización

### Cambiar horizontes temporales

**Archivo:** `quick_analysis.py` o `run_analysis.py`

```python
# Perfil Macro (línea ~51)
lot_profile_macro = build_lot_profile(df_micro, period_days=7, ...)  # Cambiar 7

# Perfil Micro (línea ~59)
lot_profile_micro = build_lot_profile(df_micro, period_days=1, ...)  # Cambiar 1

# Monte Carlo (línea ~82)
mc_simulations = run_monte_carlo(
    ...,
    num_simulations=10000,  # Cambiar cantidad de simulaciones
    num_steps=24,           # Cambiar pasos (1 paso = 1 hora)
    time_horizon=1.0        # Cambiar horizonte (1.0 = 1 día)
)
```

**Nota:** Mantener Micro alineado con Monte Carlo (ambos 1 día).

### Aumentar simulaciones Monte Carlo

Más simulaciones = mayor precisión + mayor tiempo:
- 10,000 sim: ~10 seg *(recomendado)*
- 50,000 sim: ~45 seg
- 100,000 sim: ~90 seg

---

## 🛠️ Troubleshooting

### Error: "FileNotFoundError: data/XAUUSDz_H1.csv"
**Solución:** Verificar que existan los archivos CSV en la carpeta `data/`
```powershell
ls data
```
Deberías ver: `ingresosXAUUSD40.csv`, `ingresosXAUUSD.csv`, `XAUUSDz_H1.csv`

### Error: "ModuleNotFoundError: No module named 'arch'"
**Solución:** Instalar dependencias
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### El análisis tarda mucho (>5 minutos)
**Soluciones:**
- Reducir simulaciones a 5,000 (línea ~82 en script)
- Usar `quick_analysis.py` (más rápido, sin gráficos)

### Los POCs son siempre iguales
**No es un error.** Significa convergencia temporal → Nivel ULTRA fuerte.
Divergen cuando hay movimientos fuertes recientes (breakouts, reversiones).

---

## 🔧 Instalación en Equipo Nuevo

```powershell
# 1. Clonar repositorio
git clone [URL_DEL_REPO]
cd DepthEye

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno
.\venv\Scripts\Activate.ps1

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Verificar datos
ls data  # Debe mostrar los 3 CSVs

# 6. Ejecutar análisis de prueba
python -X utf8 quick_analysis.py
```

---

## 📂 Estructura del Proyecto

```
DepthEye/
│
├── data/                           # Datos de entrada
│   ├── ingresosXAUUSD40.csv       # Microposiciones (>25 lotes)
│   ├── ingresosXAUUSD.csv         # Posiciones sync (>500 lotes)
│   └── XAUUSDz_H1.csv             # Velas OHLC H1
│
├── reports/                        # Resultados (auto-generados)
│   └── analysis_TIMESTAMP/
│
├── venv/                           # Entorno virtual Python
│
├── quick_analysis.py              # ⭐ Script principal (15 seg)
├── run_analysis.py                # Script completo con gráficos (3 min)
├── xauusd_quant_analysis.py       # Motor de análisis (646 líneas)
├── visualization.py               # Generación de gráficos (372 líneas)
├── load_data.py                   # Cargador de datos CSV
├── ayuda.py                       # Ayuda interactiva (python ayuda.py)
│
├── requirements.txt               # Dependencias Python
├── .gitignore                     # Archivos ignorados por Git
└── README.md                      # Este archivo
```

---

## 📦 Dependencias

```
pandas>=2.0.0
numpy>=1.24.0
arch>=6.2.0         # GARCH modeling
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
```

---

## 🎓 Ayuda Adicional

### Sistema de ayuda interactivo
```powershell
python ayuda.py
```

### Archivos CSV de salida

**resumen.csv:**
| Columna | Descripción |
|---------|-------------|
| precio_actual | Precio al momento del análisis |
| precio_esperado | Precio esperado en 24h (Monte Carlo) |
| volatilidad_diaria | Volatilidad GARCH (%) |
| poc_macro_7d | POC de 7 días (contexto) |
| poc_micro_24h | POC de 24h (accionable) |
| n_pois | Cantidad de POIs identificados |
| n_hlns_macro | Cantidad de HLNs Macro |
| n_hlns_micro | Cantidad de HLNs Micro |

**objetivos_probabilidades.csv:**
| Columna | Descripción |
|---------|-------------|
| Nivel | Precio del nivel (ej: $4114.87) |
| Tipo | POI, POC_Micro, POC_Macro, HLN_Micro, HLN_Macro |
| Prioridad | CRÍTICA, ALTA, MEDIA |
| Dirección | RESISTENCIA, SOPORTE |
| Prob_Toque_% | Probabilidad de toque en 24h |
| Prob_Cierre_Arriba_% | Probabilidad de cerrar por encima |
| Prob_Cierre_Abajo_% | Probabilidad de cerrar por debajo |

---

## 📝 Notas de Versión

### v2.0 (Actual)
- ✅ Segmentación Macro/Micro para alineación temporal
- ✅ Top 5 prioriza niveles accionables (Micro + POIs)
- ✅ Reporte muestra ambos POCs (contexto + acción)
- ✅ TypeError en visualizaciones corregido

### Breaking Changes desde v1.x
- CSV `resumen.csv`: columna `poc` → `poc_macro_7d` + `poc_micro_24h`
- CSV separados: `hlns_macro_7d.csv` y `hlns_micro_24h.csv`
- PNG separados: `perfil_lotes_macro_7d.png` y `perfil_lotes_micro_24h.png`

---

**Última actualización:** 10 Nov 2025  
**Autor:** DepthEye Team  
**Licencia:** Uso Personal
