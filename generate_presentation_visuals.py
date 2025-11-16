"""
Script para generar visualizaciones del proceso de entrenamiento del modelo
para la presentación de tesis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
from pathlib import Path

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Crear carpeta para las imágenes
output_dir = Path("presentation_images")
output_dir.mkdir(exist_ok=True)

# ============================================================================
# 1. PIPELINE DEL PROCESO
# ============================================================================
def create_pipeline_diagram():
    """Crea un diagrama del pipeline de entrenamiento"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    steps = [
        "1. Dataset VocalSet\n(20 cantantes profesionales\n100 muestras por vocal)",
        "2. Extracción de\nMétricas Vocales\n(11 features)",
        "3. Definición de\nGrupos de Habilidad\n(5 grupos: G1-G5)",
        "4. Generación de\nDatos Sintéticos\n(+30% con carencias)",
        "5. Entrenamiento\nXGBoost\n(5 modelos binarios)",
        "6. Evaluación\nMulti-label\n(Accuracy: 95.5%)"
    ]

    # Posiciones
    y_pos = 0.5
    x_positions = np.linspace(0.05, 0.95, len(steps))

    for i, (x, step) in enumerate(zip(x_positions, steps)):
        # Caja del paso
        bbox = dict(boxstyle='round,pad=0.8', facecolor='lightblue',
                   edgecolor='navy', linewidth=2, alpha=0.8)
        ax.text(x, y_pos, step, ha='center', va='center',
               fontsize=11, bbox=bbox, weight='bold')

        # Flecha al siguiente paso
        if i < len(steps) - 1:
            ax.annotate('', xy=(x_positions[i+1]-0.08, y_pos),
                       xytext=(x+0.08, y_pos),
                       arrowprops=dict(arrowstyle='->', lw=2.5, color='darkgreen'))

    plt.title('Pipeline de Entrenamiento del Modelo de Detección de Carencias Vocales',
             fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / '1_pipeline.png', dpi=300, bbox_inches='tight')
    print("✓ Diagrama de pipeline creado")
    plt.close()

# ============================================================================
# 2. MÉTRICAS Y GRUPOS DE HABILIDAD
# ============================================================================
def create_metrics_groups_diagram():
    """Crea un diagrama de métricas y grupos de habilidad"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    groups = {
        'G1: Soporte\nRespiratorio': [
            'meanRmsDb',
            'rmsConsistency',
            'dynamicRangeDb',
            'stabilityCents',
            'durationSec'
        ],
        'G2: Afinación\ny Oído Tonal': [
            'precisionCents',
            'stabilityCents',
            'rangeSpanSemitones',
            'meanRmsDb'
        ],
        'G3: Estabilidad\ny Vibrato': [
            'stabilityCents',
            'rmsConsistency',
            'attackLatencyMs',
            'meanRmsDb'
        ],
        'G4: Potencia\ny Control Dinámico': [
            'meanRmsDb',
            'rmsConsistency',
            'dynamicRangeDb',
            'stabilityCents'
        ],
        'G5: Rango\ny Flexibilidad': [
            'rangeMinMidi',
            'rangeMaxMidi',
            'rangeSpanSemitones',
            'stabilityCents',
            'meanRmsDb',
            'rmsConsistency'
        ]
    }

    colors = ['#FFB6C1', '#87CEEB', '#98FB98', '#FFD700', '#DDA0DD']

    y_start = 0.9
    y_step = 0.18

    for i, ((group_name, metrics), color) in enumerate(zip(groups.items(), colors)):
        y = y_start - i * y_step

        # Título del grupo
        bbox_group = dict(boxstyle='round,pad=0.6', facecolor=color,
                         edgecolor='black', linewidth=2.5, alpha=0.9)
        ax.text(0.15, y, group_name, ha='center', va='center',
               fontsize=13, bbox=bbox_group, weight='bold')

        # Métricas
        metrics_text = ' • '.join(metrics)
        bbox_metrics = dict(boxstyle='round,pad=0.5', facecolor='white',
                           edgecolor='gray', linewidth=1.5, alpha=0.8)
        ax.text(0.6, y, metrics_text, ha='center', va='center',
               fontsize=10, bbox=bbox_metrics)

        # Flecha
        ax.annotate('', xy=(0.35, y), xytext=(0.22, y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    plt.title('Grupos de Habilidad Vocal y Métricas Asociadas',
             fontsize=18, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / '2_metrics_groups.png', dpi=300, bbox_inches='tight')
    print("✓ Diagrama de métricas y grupos creado")
    plt.close()

# ============================================================================
# 3. RESULTADOS DEL MODELO
# ============================================================================
def create_results_chart():
    """Crea gráfico de barras con accuracy y F1-score"""
    # Cargar resultados
    with open('models/reports/evaluation_report.json', 'r') as f:
        results = json.load(f)

    groups = list(results.keys())
    accuracies = [results[g]['accuracy'] * 100 for g in groups]
    f1_scores = [results[g]['f1_score'] * 100 for g in groups]

    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy',
                   color='#4472C4', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score',
                   color='#ED7D31', alpha=0.8, edgecolor='black')

    # Añadir valores en las barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10, weight='bold')

    ax.set_xlabel('Grupos de Habilidad', fontsize=14, weight='bold')
    ax.set_ylabel('Porcentaje (%)', fontsize=14, weight='bold')
    ax.set_title('Rendimiento del Modelo por Grupo de Habilidad',
                fontsize=16, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])

    # Añadir línea de referencia
    ax.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.5, label='90% threshold')

    plt.tight_layout()
    plt.savefig(output_dir / '3_model_results.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico de resultados creado")
    plt.close()

# ============================================================================
# 4. MATRICES DE CONFUSIÓN
# ============================================================================
def create_confusion_matrices():
    """Crea matrices de confusión para cada grupo"""
    with open('models/reports/evaluation_report.json', 'r') as f:
        results = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    groups = list(results.keys())

    for i, group in enumerate(groups):
        cm = np.array(results[group]['confusion_matrix'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   cbar=False, ax=axes[i],
                   annot_kws={'size': 16, 'weight': 'bold'},
                   linewidths=2, linecolor='black')

        axes[i].set_title(f'{group}\nAccuracy: {results[group]["accuracy"]*100:.1f}%',
                         fontsize=13, weight='bold')
        axes[i].set_xlabel('Predicción', fontsize=11, weight='bold')
        axes[i].set_ylabel('Real', fontsize=11, weight='bold')
        axes[i].set_xticklabels(['Sin carencia', 'Con carencia'], fontsize=10)
        axes[i].set_yticklabels(['Sin carencia', 'Con carencia'], fontsize=10)

    # Remover el último subplot vacío
    fig.delaxes(axes[5])

    plt.suptitle('Matrices de Confusión por Grupo de Habilidad',
                fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / '4_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Matrices de confusión creadas")
    plt.close()

# ============================================================================
# 5. DISTRIBUCIÓN DEL DATASET
# ============================================================================
def create_dataset_distribution():
    """Crea gráficos de distribución del dataset"""
    df = pd.read_csv('data/training_data_augmented.csv')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 5.1 Distribución por género
    gender_counts = df['gender'].value_counts()
    axes[0, 0].bar(gender_counts.index, gender_counts.values,
                   color=['#FF69B4', '#4169E1'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 0].set_title('Distribución por Género', fontsize=14, weight='bold')
    axes[0, 0].set_ylabel('Cantidad de muestras', fontsize=12, weight='bold')
    for i, v in enumerate(gender_counts.values):
        axes[0, 0].text(i, v + 5, str(v), ha='center', fontsize=12, weight='bold')

    # 5.2 Distribución por vocal
    vowel_counts = df['vowel'].value_counts().sort_index()
    axes[0, 1].bar(vowel_counts.index, vowel_counts.values,
                   color='#90EE90', alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 1].set_title('Distribución por Vocal', fontsize=14, weight='bold')
    axes[0, 1].set_ylabel('Cantidad de muestras', fontsize=12, weight='bold')
    for i, v in enumerate(vowel_counts.values):
        axes[0, 1].text(i, v + 2, str(v), ha='center', fontsize=12, weight='bold')

    # 5.3 Distribución de carencias
    weak_cols = [col for col in df.columns if col.startswith('weak_')]
    weak_counts = df[weak_cols].sum().sort_values(ascending=False)
    axes[1, 0].barh(range(len(weak_counts)), weak_counts.values,
                    color='#FFD700', alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 0].set_yticks(range(len(weak_counts)))
    axes[1, 0].set_yticklabels(weak_counts.index, fontsize=11)
    axes[1, 0].set_title('Cantidad de Muestras con Carencias por Grupo',
                        fontsize=14, weight='bold')
    axes[1, 0].set_xlabel('Cantidad de muestras', fontsize=12, weight='bold')
    for i, v in enumerate(weak_counts.values):
        axes[1, 0].text(v + 1, i, str(v), va='center', fontsize=12, weight='bold')

    # 5.4 Estadísticas generales
    axes[1, 1].axis('off')
    stats_text = f"""
    ESTADÍSTICAS DEL DATASET
    
    Total de muestras: {len(df):,}
    
    Cantantes únicos: {df['singer_id'].nunique()}
    
    Géneros: {df['gender'].nunique()} (F: {(df['gender']=='F').sum()}, M: {(df['gender']=='M').sum()})
    
    Vocales: {df['vowel'].nunique()} (A, E, I, O, U)
    
    Features extraídos: 11
    
    Grupos de habilidad: 5
    
    Muestras originales: {len(df[df[weak_cols].sum(axis=1) == 0])}
    
    Muestras con carencias: {len(df[df[weak_cols].sum(axis=1) > 0])}
    
    Split: 80% Train / 20% Test
    """

    bbox = dict(boxstyle='round,pad=1', facecolor='lightyellow',
               edgecolor='orange', linewidth=3, alpha=0.9)
    axes[1, 1].text(0.5, 0.5, stats_text, ha='center', va='center',
                   fontsize=12, bbox=bbox, family='monospace', weight='bold')

    plt.suptitle('Análisis del Dataset de Entrenamiento',
                fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / '5_dataset_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Distribución del dataset creada")
    plt.close()

# ============================================================================
# 6. RESUMEN DE MÉTRICAS EXTRAÍDAS
# ============================================================================
def create_metrics_extraction_summary():
    """Crea un resumen visual de las métricas extraídas"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    metrics_info = [
        ('meanRmsDb', 'Nivel promedio de volumen', 'long_tones/straight', 'dBFS'),
        ('rmsConsistency', 'Estabilidad del volumen', 'long_tones/straight', 'dBFS (std)'),
        ('dynamicRangeDb', 'Rango dinámico', 'long_tones/messa', 'dBFS'),
        ('durationSec', 'Duración efectiva de notas', 'long_tones/straight', 'segundos'),
        ('attackLatencyMs', 'Latencia de ataque', 'long_tones/straight', 'milisegundos'),
        ('precisionCents', 'Precisión de afinación', 'scales/straight', 'cents'),
        ('stabilityCents', 'Estabilidad tonal', 'long_tones/straight', 'cents'),
        ('rangeMinMidi', 'Nota mínima', 'scales/low_piano (C)', 'MIDI'),
        ('rangeMaxMidi', 'Nota máxima', 'scales/low_piano (F)', 'MIDI'),
        ('rangeSpanSemitones', 'Extensión vocal', 'scales/low_piano', 'semitonos'),
    ]

    y_start = 0.95
    y_step = 0.09

    # Encabezado
    header_bbox = dict(boxstyle='round,pad=0.5', facecolor='navy',
                      edgecolor='black', linewidth=2, alpha=0.9)
    ax.text(0.15, y_start + 0.03, 'MÉTRICA', ha='center', va='center',
           fontsize=12, bbox=header_bbox, weight='bold', color='white')
    ax.text(0.35, y_start + 0.03, 'DESCRIPCIÓN', ha='center', va='center',
           fontsize=12, bbox=header_bbox, weight='bold', color='white')
    ax.text(0.65, y_start + 0.03, 'AUDIO UTILIZADO', ha='center', va='center',
           fontsize=12, bbox=header_bbox, weight='bold', color='white')
    ax.text(0.85, y_start + 0.03, 'UNIDAD', ha='center', va='center',
           fontsize=12, bbox=header_bbox, weight='bold', color='white')

    # Datos
    for i, (metric, desc, audio, unit) in enumerate(metrics_info):
        y = y_start - (i + 1) * y_step

        # Alternar colores
        color = '#E6F3FF' if i % 2 == 0 else '#FFF8E6'
        bbox = dict(boxstyle='round,pad=0.3', facecolor=color,
                   edgecolor='gray', linewidth=1, alpha=0.8)

        ax.text(0.15, y, metric, ha='center', va='center',
               fontsize=10, bbox=bbox, weight='bold', family='monospace')
        ax.text(0.35, y, desc, ha='center', va='center',
               fontsize=10, bbox=bbox)
        ax.text(0.65, y, audio, ha='center', va='center',
               fontsize=9, bbox=bbox, style='italic')
        ax.text(0.85, y, unit, ha='center', va='center',
               fontsize=10, bbox=bbox, weight='bold')

    plt.title('Métricas Vocales Extraídas del Dataset VocalSet',
             fontsize=18, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / '6_metrics_extraction.png', dpi=300, bbox_inches='tight')
    print("✓ Resumen de extracción de métricas creado")
    plt.close()

# ============================================================================
# EJECUTAR TODAS LAS FUNCIONES
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  GENERANDO VISUALIZACIONES PARA PRESENTACIÓN DE TESIS")
    print("="*70 + "\n")

    create_pipeline_diagram()
    create_metrics_groups_diagram()
    create_metrics_extraction_summary()
    create_results_chart()
    create_confusion_matrices()
    create_dataset_distribution()

    print("\n" + "="*70)
    print(f"  ✓ TODAS LAS IMÁGENES GUARDADAS EN: {output_dir.absolute()}")
    print("="*70 + "\n")

    print("IMÁGENES GENERADAS:")
    print("  1. 1_pipeline.png - Pipeline completo del proceso")
    print("  2. 2_metrics_groups.png - Grupos de habilidad y métricas")
    print("  3. 3_model_results.png - Accuracy y F1-Score por grupo")
    print("  4. 4_confusion_matrices.png - Matrices de confusión")
    print("  5. 5_dataset_distribution.png - Distribución del dataset")
    print("  6. 6_metrics_extraction.png - Tabla de métricas extraídas")
    print("\n✓ Listo para insertar en tu PowerPoint!\n")

