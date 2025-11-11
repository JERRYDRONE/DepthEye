"""
Módulo de Visualización para DepthEye
======================================
Herramientas para visualizar los resultados del análisis cuantitativo de XAUUSD.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Configuración de estilo
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def plot_lot_profile(lot_profile_dict: Dict, 
                     current_price: float,
                     save_path: Optional[str] = None):
    """
    Visualiza el Perfil de Lotes con POC, HLNs y LLNs marcados.
    
    Args:
        lot_profile_dict: Diccionario del perfil de lotes (output de build_lot_profile)
        current_price: Precio actual del mercado
        save_path: Ruta para guardar la imagen (opcional)
    """
    profile = lot_profile_dict.get('profile')
    if profile is None or profile.empty:
        print("⚠ No hay datos de perfil para visualizar")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Crear gráfico de barras horizontal
    prices = profile.index.values
    lots = profile.values
    
    ax.barh(prices, lots, height=0.2, color='steelblue', alpha=0.6, label='Distribución de Lotes')
    
    # Marcar POC
    poc = lot_profile_dict.get('poc')
    if poc:
        ax.axhline(y=poc, color='red', linewidth=2, linestyle='--', label=f'POC: ${poc:.2f}')
    
    # Marcar HLNs
    hlns = lot_profile_dict.get('hlns', [])
    for i, hln in enumerate(hlns[:5]):  # Top 5
        label = 'HLN (High Lotage)' if i == 0 else None
        ax.axhline(y=hln, color='orange', linewidth=1.5, linestyle=':', alpha=0.7, label=label)
    
    # Marcar LLNs
    llns = lot_profile_dict.get('llns', [])
    for i, lln in enumerate(llns[:3]):  # Top 3
        label = 'LLN (Low Lotage)' if i == 0 else None
        ax.axhline(y=lln, color='green', linewidth=1.5, linestyle='-.', alpha=0.7, label=label)
    
    # Marcar precio actual
    ax.axhline(y=current_price, color='yellow', linewidth=3, linestyle='-', label=f'Precio Actual: ${current_price:.2f}')
    
    ax.set_xlabel('Lotes Acumulados', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precio (USD)', fontsize=12, fontweight='bold')
    ax.set_title('Perfil de Lotes - XAUUSD\nDistribución de Liquidez por Nivel de Precio', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    plt.show()


def plot_monte_carlo_paths(mc_matrix: np.ndarray,
                           start_price: float,
                           sr_levels: List[float],
                           num_paths_to_plot: int = 100,
                           save_path: Optional[str] = None):
    """
    Visualiza una muestra de trayectorias de Monte Carlo con niveles S/R.
    
    Args:
        mc_matrix: Matriz de simulaciones (num_steps, num_simulations)
        start_price: Precio inicial
        sr_levels: Lista de niveles S/R a marcar
        num_paths_to_plot: Número de trayectorias a graficar
        save_path: Ruta para guardar la imagen (opcional)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    num_steps = mc_matrix.shape[0]
    time_axis = np.arange(num_steps + 1)
    
    # Seleccionar muestra aleatoria de trayectorias
    sample_indices = np.random.choice(mc_matrix.shape[1], 
                                     min(num_paths_to_plot, mc_matrix.shape[1]), 
                                     replace=False)
    
    # Graficar trayectorias individuales
    for idx in sample_indices:
        path = np.concatenate([[start_price], mc_matrix[:, idx]])
        ax.plot(time_axis, path, color='gray', alpha=0.1, linewidth=0.5)
    
    # Calcular estadísticas agregadas
    mean_path = np.concatenate([[start_price], np.mean(mc_matrix, axis=1)])
    median_path = np.concatenate([[start_price], np.median(mc_matrix, axis=1)])
    percentile_95 = np.concatenate([[start_price], np.percentile(mc_matrix, 95, axis=1)])
    percentile_5 = np.concatenate([[start_price], np.percentile(mc_matrix, 5, axis=1)])
    
    # Graficar estadísticas
    ax.plot(time_axis, mean_path, color='blue', linewidth=2, label='Media')
    ax.plot(time_axis, median_path, color='green', linewidth=2, label='Mediana', linestyle='--')
    ax.fill_between(time_axis, percentile_5, percentile_95, 
                    color='blue', alpha=0.2, label='IC 90% (P5-P95)')
    
    # Marcar precio inicial
    ax.axhline(y=start_price, color='yellow', linewidth=2, linestyle='-', label=f'Precio Inicial: ${start_price:.2f}')
    
    # Marcar niveles S/R
    for sr in sr_levels[:10]:  # Top 10 niveles
        if sr > start_price:
            color = 'red'
            alpha = 0.3
        else:
            color = 'green'
            alpha = 0.3
        ax.axhline(y=sr, color=color, linewidth=1, linestyle=':', alpha=alpha)
    
    ax.set_xlabel('Paso Temporal', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precio (USD)', fontsize=12, fontweight='bold')
    ax.set_title(f'Simulación Monte Carlo - XAUUSD\n{num_paths_to_plot} de {mc_matrix.shape[1]:,} Trayectorias', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    plt.show()


def plot_probability_heatmap(df_analysis: pd.DataFrame,
                             save_path: Optional[str] = None,
                             current_price: Optional[float] = None):
    """
    Visualiza un heatmap de probabilidades de toque para cada nivel S/R.
    
    Args:
        df_analysis: DataFrame de análisis de probabilidades
        save_path: Ruta para guardar la imagen (opcional)
        current_price: Precio actual del mercado (opcional, para compatibilidad)
    """
    if df_analysis.empty:
        print("⚠ No hay datos de análisis para visualizar")
        return
    
    # Preparar datos
    df_plot = df_analysis.copy()
    
    # Extraer valores numéricos de las columnas de probabilidad
    df_plot['Prob_Toque'] = df_plot['Prob_Toque_%'].str.rstrip('%').astype(float)
    df_plot['Prob_Cierre_Arriba'] = df_plot['Prob_Cierre_Arriba_%'].str.rstrip('%').astype(float)
    df_plot['Prob_Cierre_Abajo'] = df_plot['Prob_Cierre_Abajo_%'].str.rstrip('%').astype(float)
    
    # Seleccionar top niveles
    df_plot = df_plot.head(15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Heatmap 1: Probabilidad de Toque
    labels = [f"{row['Nivel']} ({row['Tipo']})" for _, row in df_plot.iterrows()]
    probs_touch = df_plot['Prob_Toque'].values.reshape(-1, 1)
    
    sns.heatmap(probs_touch, 
                annot=True, 
                fmt='.1f',
                cmap='RdYlGn',
                yticklabels=labels,
                xticklabels=['Prob. Toque (%)'],
                cbar_kws={'label': 'Probabilidad (%)'},
                ax=ax1,
                vmin=0,
                vmax=100)
    
    ax1.set_title('Probabilidad de Toque por Nivel S/R', fontsize=12, fontweight='bold')
    
    # Heatmap 2: Distribución de Cierre
    probs_close = df_plot[['Prob_Cierre_Arriba', 'Prob_Cierre_Abajo']].values
    
    sns.heatmap(probs_close,
                annot=True,
                fmt='.1f',
                cmap='coolwarm',
                yticklabels=labels,
                xticklabels=['Cierre Arriba (%)', 'Cierre Abajo (%)'],
                cbar_kws={'label': 'Probabilidad (%)'},
                ax=ax2,
                vmin=0,
                vmax=100)
    
    ax2.set_title('Distribución de Probabilidad de Cierre', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    plt.show()


def plot_volatility_forecast(returns: pd.Series,
                             volatility_forecast: float,
                             save_path: Optional[str] = None):
    """
    Visualiza los retornos históricos y la volatilidad pronosticada.
    
    Args:
        returns: Serie de retornos históricos
        volatility_forecast: Volatilidad pronosticada (sigma)
        save_path: Ruta para guardar la imagen (opcional)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Gráfico 1: Serie temporal de retornos
    ax1.plot(returns.index, returns.values, color='blue', alpha=0.6, linewidth=0.8)
    ax1.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax1.fill_between(returns.index, 0, returns.values, 
                     where=(returns.values > 0), color='green', alpha=0.3, label='Retornos Positivos')
    ax1.fill_between(returns.index, 0, returns.values, 
                     where=(returns.values <= 0), color='red', alpha=0.3, label='Retornos Negativos')
    
    ax1.set_ylabel('Retorno (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Serie Temporal de Retornos - XAUUSD', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Distribución de retornos
    ax2.hist(returns.dropna(), bins=50, color='steelblue', alpha=0.7, edgecolor='black', density=True)
    
    # Overlay de distribución normal
    mu = returns.mean()
    sigma = returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    ax2.plot(x, (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2),
            'r-', linewidth=2, label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')
    
    # Marcar volatilidad pronosticada
    ax2.axvline(x=volatility_forecast, color='orange', linewidth=2, linestyle='--', 
               label=f'Vol. Pronosticada: {volatility_forecast:.3f}%')
    ax2.axvline(x=-volatility_forecast, color='orange', linewidth=2, linestyle='--')
    
    ax2.set_xlabel('Retorno (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Densidad', fontsize=11, fontweight='bold')
    ax2.set_title('Distribución de Retornos y Volatilidad GARCH', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {save_path}")
    
    plt.show()


def create_comprehensive_dashboard(results: Dict,
                                   df_ohlc: pd.DataFrame,
                                   save_path: Optional[str] = None):
    """
    Crea un dashboard completo con todos los análisis.
    
    Args:
        results: Diccionario de resultados del análisis completo
        df_ohlc: DataFrame de datos OHLC
        save_path: Ruta para guardar la imagen (opcional)
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Perfil de Lotes
    ax1 = fig.add_subplot(gs[0, 0])
    profile = results['lot_profile'].get('profile')
    if profile is not None and not profile.empty:
        ax1.barh(profile.index.values, profile.values, height=0.2, color='steelblue', alpha=0.6)
        ax1.axhline(y=results['start_price'], color='yellow', linewidth=2, label='Precio Actual')
        ax1.set_xlabel('Lotes')
        ax1.set_ylabel('Precio')
        ax1.set_title('Perfil de Lotes', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. OHLC Chart
    ax2 = fig.add_subplot(gs[0, 1:])
    df_plot = df_ohlc.tail(100)
    ax2.plot(df_plot.index, df_plot['close'], color='blue', linewidth=1.5, label='Close')
    ax2.fill_between(df_plot.index, df_plot['low'], df_plot['high'], alpha=0.2, color='gray')
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Precio')
    ax2.set_title('XAUUSD - Últimas 100 Velas', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Monte Carlo Sample Paths
    ax3 = fig.add_subplot(gs[1, :])
    mc_matrix = results.get('mc_matrix')
    if mc_matrix is not None and mc_matrix.size > 0:
        sample_paths = mc_matrix[:, :50]
        for i in range(sample_paths.shape[1]):
            ax3.plot(sample_paths[:, i], color='gray', alpha=0.2, linewidth=0.5)
        mean_path = np.mean(mc_matrix, axis=1)
        ax3.plot(mean_path, color='blue', linewidth=2, label='Media')
        ax3.axhline(y=results['start_price'], color='yellow', linewidth=2, label='Inicio')
        ax3.set_xlabel('Paso Temporal')
        ax3.set_ylabel('Precio')
        ax3.set_title('Simulación Monte Carlo (50 Trayectorias)', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Tabla de Probabilidades (Top 10)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
    df_prob = results.get('probability_analysis')
    if df_prob is not None and not df_prob.empty:
        df_display = df_prob.head(10)[['Nivel', 'Tipo', 'Dirección', 'Prob_Toque_%', 'Prob_Cierre_Arriba_%']]
        table = ax4.table(cellText=df_display.values,
                         colLabels=df_display.columns,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Colorear header
        for i in range(len(df_display.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Top 10 Niveles S/R - Probabilidades de Interacción', 
                 fontweight='bold', fontsize=12, pad=20)
    
    # Título general
    fig.suptitle(f'DepthEye - Dashboard Cuantitativo XAUUSD\nPrecio: ${results["start_price"]:.2f} | Volatilidad: {results["daily_volatility"]:.4f}%',
                fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard guardado en: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Módulo de Visualización - DepthEye")
    print("Importa este módulo para usar las funciones de visualización")
    print("\nFunciones disponibles:")
    print("  - plot_lot_profile()")
    print("  - plot_monte_carlo_paths()")
    print("  - plot_probability_heatmap()")
    print("  - plot_volatility_forecast()")
    print("  - create_comprehensive_dashboard()")
