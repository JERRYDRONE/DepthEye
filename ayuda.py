"""
DepthEye - Script de Ayuda
==========================
Ejecuta este script para ver comandos útiles
"""

print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    DEPTHEYE - ANÁLISIS CUANTITATIVO XAUUSD                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 COMANDOS PRINCIPALES:

1️⃣  ACTIVAR ENTORNO VIRTUAL:
   .\\venv\\Scripts\\Activate.ps1

2️⃣  ANÁLISIS RÁPIDO (15 segundos):
   python -X utf8 quick_analysis.py

3️⃣  ANÁLISIS COMPLETO (2-3 minutos):
   python -X utf8 run_analysis.py

4️⃣  PROBAR CARGA DE DATOS:
   python -X utf8 load_data.py

5️⃣  VER ESTA AYUDA:
   python ayuda.py

────────────────────────────────────────────────────────────────────────────────

📁 ARCHIVOS DE DATOS REQUERIDOS (en carpeta data/):
   • ingresosXAUUSD40.csv   - Microposiciones (>25 lotes)
   • ingresosXAUUSD.csv     - Posiciones sync (>500 lotes)
   • XAUUSDz_H1.csv         - Datos OHLC H1

📊 REPORTES GENERADOS (en carpeta reports/):
   • resumen.csv            - Métricas principales (incluye POC Macro y Micro)
   • probabilidades.csv     - Probabilidades de toque para S/R (Macro/Micro/POIs)
   • pois.csv               - Points of Interest
   • hlns_macro_7d.csv      - High Liquidity Nodes (contexto 7 días)
   • hlns_micro_24h.csv     - High Liquidity Nodes (acción 24 horas)

────────────────────────────────────────────────────────────────────────────────

🔧 INSTALACIÓN EN EQUIPO NUEVO:
   Ver INSTALACION.md

📖 DOCUMENTACIÓN COMPLETA:
   Ver README.md

────────────────────────────────────────────────────────────────────────────────

⚠️  IMPORTANTE:
   • Siempre activa el entorno virtual antes de ejecutar
   • Usa el flag -X utf8 para evitar errores de encoding
   • Los datos CSV deben estar en la carpeta data/

╚══════════════════════════════════════════════════════════════════════════════╝
""")
