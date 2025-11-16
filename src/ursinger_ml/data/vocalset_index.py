"""
Indexación del dataset VocalSet.

Este módulo escanea la estructura de carpetas de VocalSet y crea un índice
con todos los archivos de audio disponibles.
"""
from pathlib import Path
import pandas as pd

from ursinger_ml.config import VOCALSET_ROOT

# Vocales posibles según VocalSet
VOWELS = ["a", "e", "i", "o", "u"]


def extract_vowel(filename: str):
    """
    Detecta la vocal del archivo.
    La vocal es la última letra antes de .wav

    Args:
        filename: Nombre del archivo (ej: f1_arpeggios_belt_c_a.wav)

    Returns:
        str: Vocal en mayúscula (A, E, I, O, U) o None

    Ejemplo:
        >>> extract_vowel('f1_arpeggios_belt_c_a.wav')
        'A'
    """
    # Remover la extensión .wav
    name_without_ext = filename.replace('.wav', '').replace('.WAV', '')

    # La última letra es la vocal
    if name_without_ext:
        last_char = name_without_ext[-1].lower()
        if last_char in VOWELS:
            return last_char.upper()

    return None


def extract_pitch_type(filename: str):
    """
    Detecta el tipo de pitch en archivos de escalas.
    Los archivos pueden tener _c_ (nota baja) o _f_ (nota alta).

    Args:
        filename: Nombre del archivo (ej: f1_scales_c_fast_piano_a.wav)

    Returns:
        str: 'c' (low), 'f' (high), o None si no aplica

    Ejemplo:
        >>> extract_pitch_type('f1_scales_c_fast_piano_a.wav')
        'c'
        >>> extract_pitch_type('f1_scales_f_fast_piano_a.wav')
        'f'
    """
    filename_lower = filename.lower()

    # Buscar patrón _c_ o _f_ en el nombre
    if '_c_' in filename_lower:
        return 'c'
    elif '_f_' in filename_lower:
        return 'f'

    return None


def build_vocalset_index():
    """
    Recorre el dataset VocalSet y arma un DataFrame con todos los archivos.

    Estructura de VocalSet:
        singer_id/category/technique/files.wav
        Ejemplo: female1/arpeggios/belt/f1_arpeggios_belt_c_a.wav

    Returns:
        pd.DataFrame: DataFrame con columnas:
            - singer_id: ID del cantante (ej: 'female1', 'male5')
            - gender: Género ('F' o 'M')
            - category: Categoría (ej: 'long_tones', 'scales', 'arpeggios')
            - technique: Técnica vocal (ej: 'straight', 'vibrato', 'belt')
            - vowel: Vocal cantada ('A', 'E', 'I', 'O', 'U')
            - pitch_type: Tipo de pitch ('c' para bajo, 'f' para alto, None para otros)
            - file_path: Ruta completa al archivo .wav

    Raises:
        FileNotFoundError: Si no encuentra el directorio de VocalSet
    """
    root = Path(VOCALSET_ROOT)

    if not root.exists():
        raise FileNotFoundError(f"VocalSet no encontrado en {root}")

    rows = []

    # Recorrer: singer_id/category/technique/files.wav
    for singer_dir in root.iterdir():
        if not singer_dir.is_dir():
            continue

        singer_id = singer_dir.name  # ej: female1, male1
        gender = "F" if singer_id.startswith("female") else "M"

        for category_dir in singer_dir.iterdir():
            if not category_dir.is_dir():
                continue

            category = category_dir.name  # arpeggios, scales, long_tones, excerpts

            for technique_dir in category_dir.iterdir():
                if not technique_dir.is_dir():
                    continue

                technique = technique_dir.name  # belt, breathy, vibrato, etc.

                for wav in technique_dir.glob("*.wav"):
                    filename = wav.name
                    vowel = extract_vowel(filename)
                    pitch_type = extract_pitch_type(filename)

                    rows.append({
                        "singer_id": singer_id,
                        "gender": gender,
                        "category": category,
                        "technique": technique,
                        "vowel": vowel,
                        "pitch_type": pitch_type,
                        "file_path": str(wav)
                    })

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # Configurar pandas para mostrar todas las columnas completas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    df = build_vocalset_index()
    print(df.head())
    print(f"\nTotal archivos: {len(df)}")
    print(f"Cantantes: {df['singer_id'].nunique()}")
    print(f"Categorías: {df['category'].unique().tolist()}")

