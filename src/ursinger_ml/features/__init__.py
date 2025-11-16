"""
Módulo de extracción de características de audio.

Organiza las métricas por categorías:
- acoustic.py: Métricas acústicas básicas (volumen, energía)
- pitch.py: Métricas relacionadas con el pitch
- timbre.py: Métricas de timbre y calidad vocal
"""

from .metrics_config import METRICS_CONFIG, get_audio_requirements, get_metrics_for_audio
from .acoustic import (calculate_mean_rms_db, calculate_attack_latency_ms,
                      extract_acoustic_features)
from .pitch import (calculate_precision_cents, calculate_stability_cents,
                    calculate_range_metrics, calculate_vibrato_metrics,
                    extract_pitch_features)

__all__ = [
    'METRICS_CONFIG',
    'get_audio_requirements',
    'get_metrics_for_audio',
    'calculate_mean_rms_db',
    'calculate_attack_latency_ms',
    'extract_acoustic_features',
    'calculate_precision_cents',
    'calculate_stability_cents',
    'calculate_range_metrics',
    'calculate_vibrato_metrics',
    'extract_pitch_features',
]

