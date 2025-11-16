"""
Genera datos sintéticos con carencias en grupos de habilidad.

Para el 30% de females y 30% de males:
- Por cada fila "buena", crear 5 filas "malas" con carencias en cada grupo
- Ajustar las métricas para simular debilidades realistas
- Marcar weak_G1 a weak_G5 según corresponda
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Configurar semilla para reproducibilidad
np.random.seed(42)

# Cargar dataset
df = pd.read_csv('data/training_data_with_skill_groups.csv')

print("="*70)
print("GENERANDO DATOS SINTÉTICOS CON CARENCIAS")
print("="*70)
print(f"\nDataset original: {df.shape}")

# Separar por género
df_female = df[df['gender'] == 'F'].copy()
df_male = df[df['gender'] == 'M'].copy()

print(f"Cantantes femeninas: {len(df_female)}")
print(f"Cantantes masculinos: {len(df_male)}")

# Seleccionar 30% aleatorio de cada género
n_female_sample = int(len(df_female) * 0.3)
n_male_sample = int(len(df_male) * 0.3)

print(f"\n30% de females: {n_female_sample} filas")
print(f"30% de males: {n_male_sample} filas")

# Seleccionar muestras aleatorias
female_sample = df_female.sample(n=n_female_sample, random_state=42)
male_sample = df_male.sample(n=n_male_sample, random_state=42)

sample_rows = pd.concat([female_sample, male_sample])
print(f"\nTotal de filas a expandir: {len(sample_rows)}")

# Definir cómo degradar las métricas para cada grupo
def degrade_for_G1(row):
    """G1: Soporte respiratorio y control de aire
    Métricas: meanRmsDb, rmsConsistency, dynamicRangeDb, stabilityCents, durationSec
    """
    new_row = row.copy()
    # Reducir volumen (más negativo)
    new_row['meanRmsDb'] = row['meanRmsDb'] - np.random.uniform(5, 12)
    # Aumentar inconsistencia (mayor variación)
    new_row['rmsConsistency'] = row['rmsConsistency'] + np.random.uniform(5, 10)
    # Reducir rango dinámico (menos control)
    new_row['dynamicRangeDb'] = max(20, row['dynamicRangeDb'] - np.random.uniform(15, 25))
    # Aumentar inestabilidad de pitch
    new_row['stabilityCents'] = row['stabilityCents'] + np.random.uniform(5, 12)
    # Reducir duración (menos soporte)
    new_row['durationSec'] = max(0.5, row['durationSec'] * np.random.uniform(0.4, 0.7))
    new_row['weak_G1'] = 1
    return new_row

def degrade_for_G2(row):
    """G2: Afinación y oído tonal
    Métricas: precisionCents, stabilityCents, rangeSpanSemitones, meanRmsDb
    """
    new_row = row.copy()
    # Empeorar precisión (mayor error)
    new_row['precisionCents'] = row['precisionCents'] + np.random.uniform(15, 30)
    # Aumentar inestabilidad
    new_row['stabilityCents'] = row['stabilityCents'] + np.random.uniform(8, 15)
    # Reducir rango (menos flexibilidad tonal)
    new_row['rangeSpanSemitones'] = max(8, row['rangeSpanSemitones'] - np.random.uniform(3, 7))
    new_row['rangeMinMidi'] = row['rangeMinMidi'] + np.random.uniform(1, 3)
    new_row['rangeMaxMidi'] = row['rangeMaxMidi'] - np.random.uniform(1, 3)
    # Reducir volumen ligeramente
    new_row['meanRmsDb'] = row['meanRmsDb'] - np.random.uniform(2, 5)
    new_row['weak_G2'] = 1
    return new_row

def degrade_for_G3(row):
    """G3: Estabilidad y vibrato controlado
    Métricas: stabilityCents, rmsConsistency, attackLatencyMs, meanRmsDb
    """
    new_row = row.copy()
    # Mucha inestabilidad de pitch
    new_row['stabilityCents'] = row['stabilityCents'] + np.random.uniform(10, 20)
    # Aumentar inconsistencia de volumen
    new_row['rmsConsistency'] = row['rmsConsistency'] + np.random.uniform(7, 12)
    # Aumentar latencia de ataque (menos control)
    new_row['attackLatencyMs'] = row['attackLatencyMs'] + np.random.uniform(40, 80)
    # Reducir volumen
    new_row['meanRmsDb'] = row['meanRmsDb'] - np.random.uniform(3, 8)
    new_row['weak_G3'] = 1
    return new_row

def degrade_for_G4(row):
    """G4: Potencia y control dinámico
    Métricas: meanRmsDb, rmsConsistency, dynamicRangeDb, stabilityCents
    """
    new_row = row.copy()
    # Reducir mucho el volumen (poca potencia)
    new_row['meanRmsDb'] = row['meanRmsDb'] - np.random.uniform(8, 15)
    # Aumentar inconsistencia
    new_row['rmsConsistency'] = row['rmsConsistency'] + np.random.uniform(6, 11)
    # Reducir rango dinámico
    new_row['dynamicRangeDb'] = max(25, row['dynamicRangeDb'] - np.random.uniform(10, 20))
    # Aumentar inestabilidad
    new_row['stabilityCents'] = row['stabilityCents'] + np.random.uniform(6, 12)
    new_row['weak_G4'] = 1
    return new_row

def degrade_for_G5(row):
    """G5: Rango y flexibilidad vocal
    Métricas: rangeMinMidi, rangeMaxMidi, rangeSpanSemitones, stabilityCents, meanRmsDb, rmsConsistency
    """
    new_row = row.copy()
    # Reducir significativamente el rango
    new_row['rangeMinMidi'] = row['rangeMinMidi'] + np.random.uniform(2, 5)
    new_row['rangeMaxMidi'] = row['rangeMaxMidi'] - np.random.uniform(2, 5)
    new_row['rangeSpanSemitones'] = max(6, row['rangeSpanSemitones'] - np.random.uniform(5, 10))
    # Aumentar inestabilidad
    new_row['stabilityCents'] = row['stabilityCents'] + np.random.uniform(7, 14)
    # Reducir volumen
    new_row['meanRmsDb'] = row['meanRmsDb'] - np.random.uniform(4, 9)
    # Aumentar inconsistencia
    new_row['rmsConsistency'] = row['rmsConsistency'] + np.random.uniform(5, 9)
    new_row['weak_G5'] = 1
    return new_row

# Generar filas degradadas
degradation_functions = [
    degrade_for_G1,
    degrade_for_G2,
    degrade_for_G3,
    degrade_for_G4,
    degrade_for_G5
]

new_rows = []

print("\nGenerando filas con carencias...")
for idx, row in sample_rows.iterrows():
    # Por cada fila, generar 5 versiones degradadas
    for degrade_func in degradation_functions:
        degraded_row = degrade_func(row)
        new_rows.append(degraded_row)

# Crear DataFrame con filas degradadas
df_degraded = pd.DataFrame(new_rows)

print(f"Filas degradadas generadas: {len(df_degraded)}")
print(f"  - Filas por grupo G1: {len(df_degraded[df_degraded['weak_G1'] == 1])}")
print(f"  - Filas por grupo G2: {len(df_degraded[df_degraded['weak_G2'] == 1])}")
print(f"  - Filas por grupo G3: {len(df_degraded[df_degraded['weak_G3'] == 1])}")
print(f"  - Filas por grupo G4: {len(df_degraded[df_degraded['weak_G4'] == 1])}")
print(f"  - Filas por grupo G5: {len(df_degraded[df_degraded['weak_G5'] == 1])}")

# Combinar dataset original con filas degradadas
df_final = pd.concat([df, df_degraded], ignore_index=True)

print(f"\n{'='*70}")
print(f"DATASET FINAL")
print(f"{'='*70}")
print(f"Total de filas: {len(df_final)}")
print(f"  - Filas originales (buenas): {len(df)}")
print(f"  - Filas sintéticas (con carencias): {len(df_degraded)}")

# Mostrar distribución de carencias
print(f"\nDistribución de carencias:")
for i in range(1, 6):
    col = f'weak_G{i}'
    count = len(df_final[df_final[col] == 1])
    percentage = (count / len(df_final)) * 100
    print(f"  {col}: {count} filas ({percentage:.1f}%)")

# Guardar dataset final
output_path = 'data/training_data_augmented.csv'
df_final.to_csv(output_path, index=False)

print(f"\n{'='*70}")
print(f"DATASET GUARDADO: {output_path}")
print(f"Dimensiones: {df_final.shape[0]} filas x {df_final.shape[1]} columnas")
print(f"{'='*70}")

# Mostrar ejemplos
print("\nEjemplos de filas con carencias:")
print("\nFila original (female1, A):")
original = df[(df['singer_id'] == 'female1') & (df['vowel'] == 'A')]
if len(original) > 0:
    print(original[['singer_id', 'vowel', 'meanRmsDb', 'precisionCents', 'stabilityCents', 'weak_G1', 'weak_G2', 'weak_G3', 'weak_G4', 'weak_G5']].to_string())

if len(df_degraded) > 0:
    print("\nFila con carencia en G1 (ejemplo):")
    g1_example = df_degraded[df_degraded['weak_G1'] == 1].head(1)
    print(g1_example[['singer_id', 'vowel', 'meanRmsDb', 'durationSec', 'rmsConsistency', 'weak_G1', 'weak_G2', 'weak_G3']].to_string())

    print("\nFila con carencia en G2 (ejemplo):")
    g2_example = df_degraded[df_degraded['weak_G2'] == 1].head(1)
    print(g2_example[['singer_id', 'vowel', 'precisionCents', 'stabilityCents', 'rangeSpanSemitones', 'weak_G1', 'weak_G2', 'weak_G3']].to_string())

print("\n✅ Dataset aumentado generado exitosamente!")
print(f"📊 Ratio final: {len(df)} buenas + {len(df_degraded)} malas = {len(df_final)} total")

