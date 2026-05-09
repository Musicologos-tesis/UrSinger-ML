"""
Script para generar visualizaciones del proceso de entrenamiento del modelo
para la presentacion de tesis.
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


def _load_json(path: Path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def _get_base_dataset_stats():
    base_path = Path('data/training_data.csv')
    if not base_path.exists():
        return None
    df = pd.read_csv(base_path)
    total = len(df)
    singers = df['singer_id'].nunique() if 'singer_id' in df.columns else None
    vowels = df['vowel'].nunique() if 'vowel' in df.columns else None
    per_vowel = int(round(total / vowels)) if vowels else None
    return {
        'total': total,
        'singers': singers,
        'vowels': vowels,
        'per_vowel': per_vowel
    }


def _get_augmented_count():
    aug_path = Path('data/training_data_augmented.csv')
    if not aug_path.exists():
        return None
    df = pd.read_csv(aug_path)
    return len(df)


def _get_avg_accuracy():
    report = _load_json(Path('models/reports/evaluation_report.json'))
    if not report:
        return None
    accuracies = [report[g].get('accuracy') for g in report.keys()]
    accuracies = [a for a in accuracies if a is not None]
    if not accuracies:
        return None
    return float(np.mean(accuracies))

# ============================================================================
# 1. PIPELINE DEL PROCESO
# ============================================================================
def create_pipeline_diagram():
    """Crea un diagrama del pipeline de entrenamiento"""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    base_stats = _get_base_dataset_stats()
    aug_count = _get_augmented_count()
    avg_accuracy = _get_avg_accuracy()

    if base_stats:
        per_vowel_text = f", {base_stats['per_vowel']} per vowel" if base_stats['per_vowel'] else ""
        step_1 = (
            "1. VocalSet dataset\n"
            f"({base_stats['singers']} professional singers\n"
            f"{base_stats['total']} samples total{per_vowel_text})"
        )
    else:
        step_1 = "1. VocalSet dataset\n(20 professional singers\n100 samples total)"

    if base_stats and aug_count:
        synthetic_count = max(aug_count - base_stats['total'], 0)
        step_4 = f"4. Synthetic data\ngeneration\n(+{synthetic_count} weakness samples)"
    else:
        step_4 = "4. Synthetic data\ngeneration\n(+weakness samples)"

    if avg_accuracy is not None:
        step_6 = f"6. Multi-label\nevaluation\n(Accuracy: {avg_accuracy*100:.1f}%)"
    else:
        step_6 = "6. Multi-label\nevaluation"

    steps = [
        step_1,
        "2. Vocal feature\nextraction\n(11 features)",
        "3. Skill group\ndefinition\n(5 groups: G1-G5)",
        step_4,
        "5. XGBoost\ntraining\n(5 binary models)",
        step_6
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

    plt.title('Training Pipeline for Vocal Weakness Detection',
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
        'G1: Breath\nSupport': [
            'meanRmsDb',
            'rmsConsistency',
            'dynamicRangeDb',
            'stabilityCents',
            'durationSec'
        ],
        'G2: Pitch\nAccuracy': [
            'precisionCents',
            'stabilityCents',
            'rangeSpanSemitones',
            'meanRmsDb'
        ],
        'G3: Stability\nand Vibrato': [
            'stabilityCents',
            'rmsConsistency',
            'attackLatencyMs',
            'meanRmsDb'
        ],
        'G4: Power\nand Dynamics': [
            'meanRmsDb',
            'rmsConsistency',
            'dynamicRangeDb',
            'stabilityCents'
        ],
        'G5: Range\nand Flexibility': [
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

    plt.title('Vocal Skill Groups and Associated Metrics',
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

    ax.set_xlabel('Skill Groups', fontsize=14, weight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, weight='bold')
    ax.set_title('Model Performance by Skill Group',
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
# 3B. RESULTADOS CV (BASELINES + XGBOOST)
# ============================================================================
def create_cv_results_chart():
    """Crea grafico de barras con resumen de CV por modelo"""
    report_path = Path('models/reports/cv_report.json')
    if not report_path.exists():
        print("! cv_report.json no encontrado, omitiendo grafico CV")
        return

    with open(report_path, 'r') as f:
        cv_report = json.load(f)

    summary = cv_report.get('summary', {})
    if not summary:
        print("! Resumen CV vacio, omitiendo grafico CV")
        return

    model_names = list(summary.keys())
    accuracies = [summary[m].get('avg_accuracy', 0.0) or 0.0 for m in model_names]
    f1_scores = [summary[m].get('avg_f1_score', 0.0) or 0.0 for m in model_names]
    pr_aucs = [summary[m].get('avg_pr_auc', 0.0) or 0.0 for m in model_names]

    x = np.arange(len(model_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width, np.array(accuracies) * 100, width, label='Accuracy',
                   color='#4472C4', alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x, np.array(f1_scores) * 100, width, label='F1-Score',
                   color='#ED7D31', alpha=0.85, edgecolor='black')
    bars3 = ax.bar(x + width, np.array(pr_aucs) * 100, width, label='PR-AUC',
                   color='#70AD47', alpha=0.85, edgecolor='black')

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9, weight='bold')

    ax.set_xlabel('Models', fontsize=14, weight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=14, weight='bold')
    ax.set_title('CV Results by Model (GroupKFold)',
                 fontsize=16, weight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(output_dir / '7_cv_summary.png', dpi=300, bbox_inches='tight')
    print("✓ Grafico CV creado")
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
        axes[i].set_xlabel('Prediction', fontsize=11, weight='bold')
        axes[i].set_ylabel('Actual', fontsize=11, weight='bold')
        axes[i].set_xticklabels(['No weakness', 'Weakness'], fontsize=10)
        axes[i].set_yticklabels(['No weakness', 'Weakness'], fontsize=10)

    # Remover el último subplot vacío
    fig.delaxes(axes[5])

    plt.suptitle('Confusion Matrices by Skill Group',
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
    axes[0, 0].set_title('Gender Distribution', fontsize=14, weight='bold')
    axes[0, 0].set_ylabel('Number of samples', fontsize=12, weight='bold')
    for i, v in enumerate(gender_counts.values):
        axes[0, 0].text(i, v + 5, str(v), ha='center', fontsize=12, weight='bold')

    # 5.2 Distribución por vocal
    vowel_counts = df['vowel'].value_counts().sort_index()
    axes[0, 1].bar(vowel_counts.index, vowel_counts.values,
                   color='#90EE90', alpha=0.7, edgecolor='black', linewidth=2)
    axes[0, 1].set_title('Vowel Distribution', fontsize=14, weight='bold')
    axes[0, 1].set_ylabel('Number of samples', fontsize=12, weight='bold')
    for i, v in enumerate(vowel_counts.values):
        axes[0, 1].text(i, v + 2, str(v), ha='center', fontsize=12, weight='bold')

    # 5.3 Distribución de carencias
    weak_cols = [col for col in df.columns if col.startswith('weak_')]
    weak_counts = df[weak_cols].sum().sort_values(ascending=False)
    axes[1, 0].barh(range(len(weak_counts)), weak_counts.values,
                    color='#FFD700', alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 0].set_yticks(range(len(weak_counts)))
    axes[1, 0].set_yticklabels(weak_counts.index, fontsize=11)
    axes[1, 0].set_title('Weakness Samples by Group',
                        fontsize=14, weight='bold')
    axes[1, 0].set_xlabel('Number of samples', fontsize=12, weight='bold')
    for i, v in enumerate(weak_counts.values):
        axes[1, 0].text(v + 1, i, str(v), va='center', fontsize=12, weight='bold')

    # 5.4 Estadísticas generales
    axes[1, 1].axis('off')
    stats_text = f"""
    DATASET STATISTICS
    
    Total samples: {len(df):,}
    
    Unique singers: {df['singer_id'].nunique()}
    
    Genders: {df['gender'].nunique()} (F: {(df['gender']=='F').sum()}, M: {(df['gender']=='M').sum()})
    
    Vowels: {df['vowel'].nunique()} (A, E, I, O, U)
    
    Extracted features: 11
    
    Skill groups: 5
    
    Original samples: {len(df[df[weak_cols].sum(axis=1) == 0])}
    
    Samples with weaknesses: {len(df[df[weak_cols].sum(axis=1) > 0])}
    
    Split: 80% Train / 20% Test
    """

    bbox = dict(boxstyle='round,pad=1', facecolor='lightyellow',
               edgecolor='orange', linewidth=3, alpha=0.9)
    axes[1, 1].text(0.5, 0.5, stats_text, ha='center', va='center',
                   fontsize=12, bbox=bbox, family='monospace', weight='bold')

    plt.suptitle('Training Dataset Analysis',
                fontsize=18, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / '5_dataset_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Dataset distribution created")
    plt.close()

# ============================================================================
# 6. RESUMEN DE MÉTRICAS EXTRAÍDAS
# ============================================================================
def create_metrics_extraction_summary():
    """Crea un resumen visual de las métricas extraídas"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    metrics_info = [
        ('meanRmsDb', 'Average volume level', 'long_tones/straight', 'dBFS'),
        ('rmsConsistency', 'Volume stability', 'long_tones/straight', 'dBFS (std)'),
        ('dynamicRangeDb', 'Dynamic range', 'long_tones/messa', 'dBFS'),
        ('durationSec', 'Effective note duration', 'long_tones/straight', 'seconds'),
        ('attackLatencyMs', 'Attack latency', 'long_tones/straight', 'milliseconds'),
        ('precisionCents', 'Pitch accuracy', 'scales/straight', 'cents'),
        ('stabilityCents', 'Pitch stability', 'long_tones/straight', 'cents'),
        ('rangeMinMidi', 'Minimum note', 'scales/fast_piano (C)', 'MIDI'),
        ('rangeMaxMidi', 'Maximum note', 'scales/fast_piano (F)', 'MIDI'),
        ('rangeSpanSemitones', 'Vocal range span', 'scales/fast_piano', 'semitones'),
    ]

    y_start = 0.95
    y_step = 0.09

    # Header
    header_bbox = dict(boxstyle='round,pad=0.5', facecolor='navy',
                      edgecolor='black', linewidth=2, alpha=0.9)
    ax.text(0.15, y_start + 0.03, 'METRIC', ha='center', va='center',
           fontsize=12, bbox=header_bbox, weight='bold', color='white')
    ax.text(0.35, y_start + 0.03, 'DESCRIPTION', ha='center', va='center',
           fontsize=12, bbox=header_bbox, weight='bold', color='white')
    ax.text(0.65, y_start + 0.03, 'AUDIO SOURCE', ha='center', va='center',
           fontsize=12, bbox=header_bbox, weight='bold', color='white')
    ax.text(0.85, y_start + 0.03, 'UNIT', ha='center', va='center',
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

    plt.title('Extracted Vocal Metrics from VocalSet Dataset',
             fontsize=18, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / '6_metrics_extraction.png', dpi=300, bbox_inches='tight')
    print("✓ Metrics extraction summary created")
    plt.close()

# ============================================================================
# EJECUTAR TODAS LAS FUNCIONES
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("  GENERATING VISUALS FOR THESIS PRESENTATION")
    print("="*70 + "\n")

    create_pipeline_diagram()
    create_metrics_groups_diagram()
    create_metrics_extraction_summary()
    create_results_chart()
    create_cv_results_chart()
    create_confusion_matrices()
    create_dataset_distribution()

    print("\n" + "="*70)
    print(f"  ✓ TODAS LAS IMÁGENES GUARDADAS EN: {output_dir.absolute()}")
    print("="*70 + "\n")

    print("GENERATED IMAGES:")
    print("  1. 1_pipeline.png - Full pipeline overview")
    print("  2. 2_metrics_groups.png - Skill groups and metrics")
    print("  3. 3_model_results.png - Accuracy and F1 by group")
    print("  7. 7_cv_summary.png - CV summary by model")
    print("  4. 4_confusion_matrices.png - Confusion matrices")
    print("  5. 5_dataset_distribution.png - Dataset distribution")
    print("  6. 6_metrics_extraction.png - Metrics extraction table")
    print("\n✓ Ready to insert into your presentation!\n")

