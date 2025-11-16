"""
Crea un dataset con grupos de habilidad vocal.

Elimina las métricas de vibrato y agrega 5 columnas de grupos de habilidad:
- weak_G1: Soporte respiratorio y control de aire
- weak_G2: Afinación y oído tonal
- weak_G3: Estabilidad y vibrato controlado
- weak_G4: Potencia y control dinámico
- weak_G5: Rango y flexibilidad vocal

Para cantantes de VocalSet (experimentados), todos los valores son 0.
"""
import pandas as pd
from pathlib import Path

# Cargar dataset original
df = pd.read_csv('data/training_data.csv')

print("="*70)
print("CREANDO DATASET CON GRUPOS DE HABILIDAD")
print("="*70)
print(f"\nDataset original: {df.shape}")
print(f"Columnas: {df.columns.tolist()}")

# Eliminar métricas de vibrato
columns_to_drop = ['vibratoRateHz', 'vibratoDepthCents']
df_no_vibrato = df.drop(columns=columns_to_drop)

print(f"\nDespués de eliminar vibrato: {df_no_vibrato.shape}")
print(f"Columnas eliminadas: {columns_to_drop}")

# Agregar columnas de grupos de habilidad
# Para cantantes de VocalSet (experimentados), todos los valores son 0
df_no_vibrato['weak_G1'] = 0  # Soporte respiratorio y control de aire
df_no_vibrato['weak_G2'] = 0  # Afinación y oído tonal
df_no_vibrato['weak_G3'] = 0  # Estabilidad y vibrato controlado
df_no_vibrato['weak_G4'] = 0  # Potencia y control dinámico
df_no_vibrato['weak_G5'] = 0  # Rango y flexibilidad vocal

print(f"\nDespués de agregar grupos de habilidad: {df_no_vibrato.shape}")
print(f"\nColumnas finales: {df_no_vibrato.columns.tolist()}")

# Guardar nuevo dataset
output_path = 'data/training_data_with_skill_groups.csv'
df_no_vibrato.to_csv(output_path, index=False)

print(f"\n{'='*70}")
print(f"DATASET GUARDADO: {output_path}")
print(f"{'='*70}")

# Mostrar estadísticas
print(f"\nDimensiones finales: {df_no_vibrato.shape[0]} filas x {df_no_vibrato.shape[1]} columnas")

print("\nPrimeras 5 filas:")
print(df_no_vibrato.head())

print("\n\nGRUPOS DE HABILIDAD Y SUS MÉTRICAS:")
print("="*70)

groups = {
    "G1 - Soporte respiratorio y control de aire": [
        "meanRmsDb", "rmsConsistency", "dynamicRangeDb",
        "stabilityCents", "durationSec"
    ],
    "G2 - Afinación y oído tonal": [
        "precisionCents", "stabilityCents", "rangeSpanSemitones", "meanRmsDb"
    ],
    "G3 - Estabilidad y vibrato controlado": [
        "stabilityCents", "rmsConsistency", "attackLatencyMs", "meanRmsDb"
    ],
    "G4 - Potencia y control dinámico": [
        "meanRmsDb", "rmsConsistency", "dynamicRangeDb", "stabilityCents"
    ],
    "G5 - Rango y flexibilidad vocal": [
        "rangeMinMidi", "rangeMaxMidi", "rangeSpanSemitones",
        "stabilityCents", "meanRmsDb", "rmsConsistency"
    ]
}

for group_name, metrics in groups.items():
    print(f"\n{group_name}:")
    for metric in metrics:
        print(f"  - {metric}")

print("\n" + "="*70)
print("VALORES DE weak_G1 a weak_G5:")
print("="*70)
print("Todos los cantantes de VocalSet tienen valor 0 (experimentados)")
print(df_no_vibrato[['singer_id', 'vowel', 'weak_G1', 'weak_G2', 'weak_G3', 'weak_G4', 'weak_G5']].head(10))

print("\n✅ Dataset creado exitosamente!")

