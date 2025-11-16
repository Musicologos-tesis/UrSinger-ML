"""
Construcción del dataset de entrenamiento.

Este módulo orquesta la extracción de características de todos los archivos
de audio y genera un dataset listo para entrenar modelos de Machine Learning.
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from ursinger_ml.data import build_vocalset_index
from ursinger_ml.features.metrics_config import get_audio_requirements
from ursinger_ml.features.acoustic import extract_acoustic_features
from ursinger_ml.features.pitch import extract_pitch_features
from ursinger_ml.features.timbre import extract_timbre_features


def extract_all_features(audio_files: dict):
    """
    Extrae TODAS las características de los archivos de audio proporcionados.

    Combina características de:
    - Acústicas (volumen, energía)
    - Pitch (vibrato, estabilidad)
    - Timbre (MFCC, color vocal)

    Args:
        audio_files: Diccionario que mapea (category, technique) -> file_path
                     Ejemplo: {
                         ('long_tones', 'straight'): '/path/to/file1.wav',
                         ('long_tones', 'vibrato'): '/path/to/file2.wav',
                     }

    Returns:
        dict: Diccionario con todas las métricas calculadas
    """
    features = {}

    # Extraer características acústicas
    acoustic_features = extract_acoustic_features(audio_files)
    features.update(acoustic_features)

    # Extraer características de pitch
    pitch_features = extract_pitch_features(audio_files)
    features.update(pitch_features)

    # Extraer características de timbre
    timbre_features = extract_timbre_features(audio_files)
    features.update(timbre_features)

    return features


def build_training_dataset():
    """
    Construye el dataset de entrenamiento con todas las métricas.

    Proceso:
    1. Indexa todos los archivos de VocalSet
    2. Determina qué archivos necesita según metrics_config.py
    3. Agrupa archivos por cantante y vocal
    4. Calcula todas las métricas para cada combinación
    5. Retorna un DataFrame listo para entrenar

    Returns:
        pd.DataFrame: Dataset con columnas [singer_id, gender, vowel, métricas...]
                      Una fila por cada combinación de cantante + vocal
    """
    # 1. Obtener el índice completo del dataset
    print("Indexando VocalSet...")
    df_index = build_vocalset_index()

    # 2. Obtener los requisitos de audio desde la configuración
    audio_requirements = get_audio_requirements()
    print(f"\nArchivos de audio requeridos por las metricas:")
    for category, technique in sorted(audio_requirements):
        print(f"  - {category}/{technique}")

    # 3. Filtrar el índice para obtener solo los archivos necesarios
    df_filtered = df_index[
        df_index.apply(
            lambda row: (row['category'], row['technique']) in audio_requirements,
            axis=1
        )
    ].copy()

    print(f"\nArchivos filtrados: {len(df_filtered)}")

    # 4. Agrupar por cantante y vocal para obtener todos los archivos necesarios
    print("\nAgrupando archivos por cantante y vocal...")
    grouped = df_filtered.groupby(['singer_id', 'gender', 'vowel'])

    print(f"Combinaciones unicas de (cantante, vocal): {len(grouped)}")

    # 5. Calcular métricas para cada combinación de cantante + vocal
    print("\nCalculando metricas de audio...")

    rows = []
    for (singer_id, gender, vowel), group in tqdm(grouped, desc="Procesando"):
        try:
            # Crear diccionario de archivos de audio disponibles para este cantante+vocal
            audio_files = {}
            for _, row in group.iterrows():
                # Para archivos con pitch_type (escalas _c_ y _f_), usar clave de 3 elementos
                if pd.notna(row['pitch_type']):
                    key = (row['category'], row['technique'], row['pitch_type'])
                else:
                    key = (row['category'], row['technique'])
                audio_files[key] = row['file_path']

            # Verificar archivos requeridos (sin pitch_type para la verificación general)
            missing_files = audio_requirements - set((cat, tech) for cat, tech in audio_requirements)

            # Para métricas de rango, verificar que existan ambos archivos _c_ y _f_
            if ('scales', 'fast_piano') in audio_requirements:
                key_c = ('scales', 'fast_piano', 'c')
                key_f = ('scales', 'fast_piano', 'f')

                # Buscar archivos _c_ y _f_ si no están
                if key_c not in audio_files or key_f not in audio_files:
                    # Buscar en el índice filtrado
                    for pitch_type in ['c', 'f']:
                        key = ('scales', 'fast_piano', pitch_type)
                        if key not in audio_files:
                            fallback = df_filtered[
                                (df_filtered['singer_id'] == singer_id) &
                                (df_filtered['category'] == 'scales') &
                                (df_filtered['technique'] == 'fast_piano') &
                                (df_filtered['pitch_type'] == pitch_type)
                            ]

                            if not fallback.empty:
                                fallback_file = fallback.iloc[0]['file_path']
                                fallback_vowel = fallback.iloc[0]['vowel']
                                audio_files[key] = fallback_file
                                if fallback_vowel != vowel:
                                    print(f"\nInfo: {singer_id} vocal {vowel} - usando scales/fast_piano/{pitch_type} de vocal {fallback_vowel}")

            # Verificar otros archivos necesarios (método antiguo)
            simple_missing = []
            for cat, tech in audio_requirements:
                # Verificar si existe con o sin pitch_type
                has_file = False
                if (cat, tech) in audio_files:
                    has_file = True
                else:
                    # Verificar si existe alguna versión con pitch_type
                    for key in audio_files.keys():
                        if len(key) >= 2 and key[0] == cat and key[1] == tech:
                            has_file = True
                            break

                if not has_file:
                    simple_missing.append((cat, tech))

            if simple_missing:
                # Intentar buscar archivos de respaldo
                files_found = []
                for missing_category, missing_technique in simple_missing:
                    fallback = df_filtered[
                        (df_filtered['singer_id'] == singer_id) &
                        (df_filtered['category'] == missing_category) &
                        (df_filtered['technique'] == missing_technique)
                    ]

                    if not fallback.empty:
                        fallback_file = fallback.iloc[0]['file_path']
                        fallback_vowel = fallback.iloc[0]['vowel']
                        audio_files[(missing_category, missing_technique)] = fallback_file
                        files_found.append((missing_category, missing_technique))
                        print(f"\nInfo: {singer_id} vocal {vowel} - usando {missing_category}/{missing_technique} de vocal {fallback_vowel} como respaldo")

                simple_missing = [m for m in simple_missing if m not in files_found]

            if simple_missing:
                print(f"\nAdvertencia: {singer_id} vocal {vowel} - no hay archivos disponibles para: {simple_missing}")
                continue

            # Extraer todas las características usando todos los archivos necesarios
            features = extract_all_features(audio_files)

            # Crear fila con toda la información
            data_row = {
                'singer_id': singer_id,
                'gender': gender,
                'vowel': vowel,
                **features  # Agregar todas las métricas calculadas
            }

            rows.append(data_row)

        except Exception as e:
            print(f"\nError procesando {singer_id} vocal {vowel}: {e}")
            continue

    # 6. Crear DataFrame final
    df_training = pd.DataFrame(rows)

    # 7. Ordenar por cantante y vocal
    df_training = df_training.sort_values(['singer_id', 'vowel']).reset_index(drop=True)

    return df_training


def save_training_dataset(df: pd.DataFrame, output_path: str = None):
    """
    Guarda el dataset de entrenamiento en un archivo CSV.

    Args:
        df: DataFrame con el dataset
        output_path: Ruta donde guardar (por defecto: data/training_data.csv)
    """
    if output_path is None:
        from ursinger_ml.config import DATA_DIR
        output_path = DATA_DIR / "training_data.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\nDataset guardado en: {output_path}")
    print(f"Tamaño: {len(df)} filas x {len(df.columns)} columnas")


if __name__ == "__main__":
    # Configurar pandas para mejor visualización
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

    # Construir dataset
    df = build_training_dataset()

    # Mostrar resumen
    print("\n" + "="*60)
    print("📊 DATASET DE ENTRENAMIENTO")
    print("="*60)
    print(f"\n🔝 Primeras filas:")
    print(df.head(10))

    print(f"\n\n🔽 Últimas filas:")
    print(df.tail(5))

    if 'meanRmsDb' in df.columns:
        print(f"\n\n📈 Estadísticas de meanRmsDb:")
        print(df['meanRmsDb'].describe())

    print(f"\n\n👥 Distribución por cantante:")
    print(df.groupby('singer_id').size())

    # Guardar dataset
    save_training_dataset(df)

