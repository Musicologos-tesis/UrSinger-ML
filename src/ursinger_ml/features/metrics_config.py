"""
Configuración de métricas y sus requisitos de archivos de audio.

Define qué archivos de audio necesita cada métrica para ser calculada.
"""

# Definición de métricas y sus requisitos de audio
# Cada métrica especifica:
# - name: nombre de la métrica
# - audio_sources: lista de diccionarios con category y technique necesarios
# - description: descripción de qué mide la métrica

METRICS_CONFIG = [
    {
        "name": "meanRmsDb",
        "description": "Nivel promedio de volumen durante la emisión (dBFS)",
        "audio_sources": [
            {"category": "long_tones", "technique": "straight"}
        ]
    },
    {
        "name": "rmsConsistency",
        "description": "Estabilidad del volumen - flujo de aire y soporte (dBFS)",
        "audio_sources": [
            {"category": "long_tones", "technique": "straight"}
        ]
    },
    {
        "name": "dynamicRangeDb",
        "description": "Diferencia entre el volumen más bajo y más alto detectado (dBFS)",
        "audio_sources": [
            {"category": "long_tones", "technique": "messa"}
        ]
    },
    {
        "name": "durationSec",
        "description": "Duración efectiva promedio de las notas sostenidas (segundos)",
        "audio_sources": [
            {"category": "long_tones", "technique": "straight"}
        ]
    },
    {
        "name": "precisionCents",
        "description": "Diferencia promedio entre el pitch emitido y el objetivo (cents)",
        "audio_sources": [
            {"category": "scales", "technique": "straight"}
        ]
    },
    {
        "name": "stabilityCents",
        "description": "Desviación tonal durante notas sostenidas (cents)",
        "audio_sources": [
            {"category": "long_tones", "technique": "straight"}
        ]
    },
    {
        "name": "rangeMinMidi",
        "description": "Nota mínima del rango vocal (MIDI)",
        "audio_sources": [
            {"category": "scales", "technique": "fast_piano"}
        ]
    },
    {
        "name": "rangeMaxMidi",
        "description": "Nota máxima del rango vocal (MIDI)",
        "audio_sources": [
            {"category": "scales", "technique": "fast_piano"}
        ]
    },
    {
        "name": "rangeSpanSemitones",
        "description": "Amplitud del rango vocal en semitonos",
        "audio_sources": [
            {"category": "scales", "technique": "fast_piano"}
        ]
    },
    {
        "name": "vibratoRateHz",
        "description": "Frecuencia de las oscilaciones de tono del vibrato (Hz)",
        "audio_sources": [
            {"category": "long_tones", "technique": "forte"}
        ]
    },
    {
        "name": "vibratoDepthCents",
        "description": "Amplitud de las oscilaciones del vibrato (cents)",
        "audio_sources": [
            {"category": "long_tones", "technique": "forte"}
        ]
    },
    {
        "name": "attackLatencyMs",
        "description": "Tiempo entre inicio de sonido y estabilización del tono (ms)",
        "audio_sources": [
            {"category": "long_tones", "technique": "straight"}
        ]
    },
    # Ejemplo de métrica que podría usar múltiples fuentes:
    # {
    #     "name": "vibrato_rate",
    #     "description": "Velocidad de vibrato en Hz",
    #     "audio_sources": [
    #         {"category": "long_tones", "technique": "vibrato"}
    #     ]
    # },
    # Ejemplo de métrica que usa dos archivos diferentes:
    # {
    #     "name": "dynamic_range",
    #     "description": "Rango dinámico comparando forte vs piano",
    #     "audio_sources": [
    #         {"category": "long_tones", "technique": "forte"},
    #         {"category": "long_tones", "technique": "pp"}
    #     ]
    # },
]


def get_audio_requirements():
    """
    Retorna un set de todas las combinaciones únicas de (category, technique)
    necesarias para calcular todas las métricas configuradas.

    Returns:
        set: Conjunto de tuplas (category, technique)
    """
    requirements = set()
    for metric in METRICS_CONFIG:
        for source in metric["audio_sources"]:
            requirements.add((source["category"], source["technique"]))
    return requirements


def get_metrics_for_audio(category: str, technique: str):
    """
    Retorna las métricas que usan un archivo específico de audio.

    Args:
        category: Categoría del audio (ej: 'long_tones')
        technique: Técnica del audio (ej: 'straight')

    Returns:
        list: Lista de nombres de métricas que usan este audio
    """
    metrics = []
    for metric in METRICS_CONFIG:
        for source in metric["audio_sources"]:
            if source["category"] == category and source["technique"] == technique:
                metrics.append(metric["name"])
                break
    return metrics


if __name__ == "__main__":
    print("Configuración de métricas:")
    print("=" * 60)
    for metric in METRICS_CONFIG:
        print(f"\n{metric['name']}")
        print(f"  Descripción: {metric['description']}")
        print(f"  Archivos necesarios:")
        for source in metric['audio_sources']:
            print(f"    - {source['category']}/{source['technique']}")

    print("\n\nRequisitos de audio únicos:")
    print("=" * 60)
    for category, technique in sorted(get_audio_requirements()):
        print(f"  - {category}/{technique}")

