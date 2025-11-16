"""
Métricas acústicas básicas de audio.

Este módulo contiene funciones para calcular características acústicas
como volumen, energía, intensidad, etc.
"""
import numpy as np
import librosa


def calculate_mean_rms_db(audio_path: str, sr: int = 22050):
    """
    Calcula el nivel promedio de volumen durante la emisión.

    Args:
        audio_path: Ruta al archivo de audio .wav
        sr: Sample rate (por defecto 22050 Hz)

    Returns:
        float: Nivel promedio de volumen en dBFS (decibels Full Scale)

    Descripción:
        El RMS (Root Mean Square) mide la energía promedio de la señal.
        dBFS es la escala donde 0 dB es el nivel máximo posible.
        Valores típicos: -40 dB (muy suave) a -10 dB (muy fuerte)

    Ejemplo:
        >>> rms = calculate_mean_rms_db('audio.wav')
        >>> print(f"Volumen: {rms:.2f} dBFS")
        Volumen: -25.79 dBFS
    """
    # Cargar el audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Calcular RMS (Root Mean Square)
    rms = librosa.feature.rms(y=y)[0]

    # Calcular el promedio del RMS
    mean_rms = np.mean(rms)

    # Convertir a dBFS (decibels Full Scale)
    # Evitar log de 0 añadiendo un valor muy pequeño
    mean_rms_db = 20 * np.log10(mean_rms + 1e-10)

    return float(mean_rms_db)


def calculate_rms_consistency(audio_path: str, sr: int = 22050):
    """
    Calcula la estabilidad del volumen (flujo de aire y soporte).

    Args:
        audio_path: Ruta al archivo de audio .wav
        sr: Sample rate (por defecto 22050 Hz)

    Returns:
        float: Estabilidad del volumen en dBFS (desviación estándar)

    Descripción:
        Mide qué tan consistente es el volumen durante la emisión.
        Un valor bajo indica mayor consistencia (mejor soporte/control).
        Un valor alto indica fluctuaciones (menos control).

        Se calcula como la desviación estándar del RMS en dB.
        Valores típicos: 1-3 dB (excelente), 3-6 dB (bueno), >6 dB (inestable)

    Ejemplo:
        >>> consistency = calculate_rms_consistency('audio.wav')
        >>> print(f"Consistencia: {consistency:.2f} dB")
        Consistencia: 2.15 dB
    """
    # Cargar el audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Calcular RMS (Root Mean Square)
    rms = librosa.feature.rms(y=y)[0]

    # Convertir cada frame de RMS a dBFS
    rms_db = 20 * np.log10(rms + 1e-10)

    # Calcular la desviación estándar (consistencia)
    # Menor desviación = más consistente
    rms_std = np.std(rms_db)

    return float(rms_std)


def calculate_dynamic_range_db(audio_path: str, sr: int = 22050):
    """
    Calcula el rango dinámico (diferencia entre volumen más bajo y más alto).

    Args:
        audio_path: Ruta al archivo de audio .wav (idealmente messa di voce)
        sr: Sample rate (por defecto 22050 Hz)

    Returns:
        float: Rango dinámico en dB

    Descripción:
        Mide la capacidad del cantante para variar el volumen.
        Especialmente útil con técnica "messa di voce" (suave → fuerte → suave).

        Se calcula como la diferencia entre el RMS máximo y mínimo en dB.
        Valores típicos:
        - 10-15 dB: Rango moderado
        - 15-25 dB: Buen rango dinámico
        - >25 dB: Excelente control dinámico

    Ejemplo:
        >>> dynamic_range = calculate_dynamic_range_db('messa_a.wav')
        >>> print(f"Rango dinámico: {dynamic_range:.2f} dB")
        Rango dinámico: 22.45 dB
    """
    # Cargar el audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Calcular RMS (Root Mean Square)
    rms = librosa.feature.rms(y=y)[0]

    # Convertir a dBFS
    rms_db = 20 * np.log10(rms + 1e-10)

    # Calcular rango dinámico: diferencia entre máximo y mínimo
    max_db = np.max(rms_db)
    min_db = np.min(rms_db)
    dynamic_range = max_db - min_db

    return float(dynamic_range)


def calculate_duration_sec(audio_path: str, sr: int = 22050, top_n: int = 3):
    """
    Calcula la duración efectiva promedio de las notas sostenidas.

    Args:
        audio_path: Ruta al archivo de audio .wav (long_tones/straight)
        sr: Sample rate (por defecto 22050 Hz)
        top_n: Número de notas a considerar (por defecto 3)

    Returns:
        float: Duración promedio de las notas sostenidas en segundos

    Descripción:
        En archivos de long_tones/straight, el cantante sostiene 3 notas
        con o sin pausas entre ellas. Esta función:
        1. Usa CREPE para detectar el pitch en cada momento
        2. Identifica cambios bruscos de pitch (cambio de nota)
        3. Mide la duración de cada segmento de pitch estable
        4. Calcula el promedio de las 3 notas más largas

        Esto mide el soporte respiratorio y capacidad de sostener notas.
        Valores típicos: 2-4 segundos (principiante), 4-8 segundos (intermedio),
                        >8 segundos (avanzado)

    Ejemplo:
        >>> duration = calculate_duration_sec('straight_a.wav')
        >>> print(f"Duración promedio: {duration:.2f} segundos")
        Duración promedio: 5.23 segundos
    """
    import crepe

    # Cargar el audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Usar CREPE para detectar pitch
    # step_size más grande = más rápido pero menos preciso (50ms = 20 frames/segundo)
    time, frequency, confidence, activation = crepe.predict(
        y, sr,
        viterbi=True,
        step_size=50,  # 50ms en lugar de 10ms para ser más rápido
        verbose=0,
        model_capacity='tiny'  # Modelo pequeño para mayor velocidad
    )

    # Filtrar por confianza (solo considerar pitch confiable)
    conf_threshold = 0.5
    valid_pitch = confidence > conf_threshold

    segments = []

    if np.sum(valid_pitch) > 3:  # Al menos 3 frames válidos
        # Convertir Hz a semitonos (log scale)
        pitch_hz = frequency[valid_pitch]
        time_valid = time[valid_pitch]

        # Evitar log de 0
        pitch_semitones = 12 * np.log2((pitch_hz + 1e-10) / 440.0)

        # Detectar cambios bruscos (más de 1.5 semitonos = cambio de nota)
        pitch_diff = np.abs(np.diff(pitch_semitones))
        note_changes = pitch_diff > 1.5  # Umbral de 1.5 semitonos

        # Encontrar índices de cambio
        change_indices = np.concatenate([[0], np.where(note_changes)[0] + 1, [len(time_valid)]])

        # Calcular duración de cada segmento
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]

            if end_idx > start_idx:
                duration = time_valid[end_idx - 1] - time_valid[start_idx]

                # Solo considerar segmentos de al menos 0.5 segundos
                if duration >= 0.5:
                    segments.append(duration)

    # Si no se detectaron suficientes segmentos con CREPE, usar RMS como respaldo
    if len(segments) < 2:
        # Calcular RMS para detectar actividad
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        # Umbral para detectar actividad
        threshold = np.mean(rms) * 0.3
        active_frames = rms > threshold

        # Convertir frames a tiempo
        times = librosa.frames_to_time(np.arange(len(active_frames)), sr=sr, hop_length=hop_length)

        # Encontrar segmentos continuos
        in_segment = False
        start_time = 0

        for i, is_active in enumerate(active_frames):
            if is_active and not in_segment:
                start_time = times[i]
                in_segment = True
            elif not is_active and in_segment:
                duration = times[i] - start_time
                if duration >= 0.5:
                    segments.append(duration)
                in_segment = False

        if in_segment and len(times) > 0:
            duration = times[-1] - start_time
            if duration >= 0.5:
                segments.append(duration)

    # Ordenar y tomar las top_n más largas
    segments.sort(reverse=True)
    top_segments = segments[:min(top_n, len(segments))]

    # Calcular promedio
    avg_duration = np.mean(top_segments) if top_segments else 0.0

    return float(avg_duration)


def calculate_attack_latency_ms(audio_path: str, sr: int = 22050):
    """
    Calcula el tiempo de ataque (latencia entre inicio de sonido y estabilización del tono).

    Args:
        audio_path: Ruta al archivo de audio .wav (long_tones/straight)
        sr: Sample rate (por defecto 22050 Hz)

    Returns:
        float: Tiempo promedio de ataque en milisegundos

    Descripción:
        En archivos de long_tones/straight, el cantante sostiene 3 notas.
        Para cada nota:
        1. Detecta el onset (inicio del sonido) usando energía
        2. Detecta cuando el pitch se estabiliza usando CREPE
        3. Calcula el tiempo entre ambos eventos
        4. Promedia los tiempos de las 3 notas

        Esto mide la precisión del ataque vocal.
        Valores típicos: 20-50 ms (rápido), 50-100 ms (normal), >100 ms (lento)

    Ejemplo:
        >>> latency = calculate_attack_latency_ms('straight_a.wav')
        >>> print(f"Attack latency: {latency:.1f} ms")
        Attack latency: 45.2 ms
    """
    import crepe

    # Cargar el audio
    y, sr = librosa.load(audio_path, sr=sr)

    # 1. Detectar onsets (inicios de sonido) usando energía
    hop_length = 512
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length,
        backtrack=True,  # Refinar la detección
        units='frames'
    )

    # Convertir frames a tiempo
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    # 2. Usar CREPE para detectar pitch
    time_pitch, frequency, confidence, _ = crepe.predict(
        y, sr,
        viterbi=True,
        step_size=10,  # 10ms para buena resolución temporal
        verbose=0,
        model_capacity='tiny'
    )

    # Filtrar por confianza
    conf_threshold = 0.6
    valid_mask = confidence > conf_threshold

    if np.sum(valid_mask) < 5 or len(onset_times) < 1:
        return 0.0

    frequency_valid = frequency[valid_mask]
    time_valid = time_pitch[valid_mask]

    # Convertir a MIDI
    midi_values = 12 * np.log2(frequency_valid / 440.0 + 1e-10) + 69

    # 3. Detectar segmentos de notas (cambios bruscos de pitch)
    midi_diff = np.abs(np.diff(midi_values))
    note_changes = midi_diff > 0.8

    # Índices de cambio
    change_indices = np.concatenate([[0], np.where(note_changes)[0] + 1, [len(midi_values)]])

    # 4. Calcular latencia de ataque para cada nota
    latencies = []

    for i in range(len(change_indices) - 1):
        start_idx = change_indices[i]
        end_idx = change_indices[i + 1]

        segment_length = end_idx - start_idx

        # Solo procesar segmentos con al menos 10 frames
        if segment_length < 10:
            continue

        segment_time = time_valid[start_idx:end_idx]
        segment_midi = midi_values[start_idx:end_idx]

        if len(segment_time) < 10:
            continue

        # Tiempo de inicio del segmento
        segment_start_time = segment_time[0]

        # Encontrar el onset más cercano antes del inicio del segmento
        onset_before = onset_times[onset_times <= segment_start_time + 0.1]

        if len(onset_before) == 0:
            # Si no hay onset antes, usar el tiempo de inicio del segmento
            onset_time = segment_start_time
        else:
            onset_time = onset_before[-1]

        # Detectar estabilización del pitch
        # Calcular la desviación estándar móvil del pitch
        window_size = min(5, len(segment_midi) // 3)

        if window_size < 2:
            continue

        # Buscar el punto donde el pitch se estabiliza
        # (desviación estándar < umbral durante ventana)
        stability_threshold = 0.15  # ~15 cents

        for j in range(window_size, len(segment_midi)):
            window = segment_midi[j-window_size:j]
            std = np.std(window)

            if std < stability_threshold:
                # Pitch estable encontrado
                stable_time = segment_time[j]

                # Calcular latencia en milisegundos
                latency_ms = (stable_time - onset_time) * 1000

                # Filtrar valores poco realistas (0-500 ms)
                if 0 <= latency_ms <= 500:
                    latencies.append(latency_ms)
                break

    # Promedio de latencias
    avg_latency = np.mean(latencies) if latencies else 0.0

    return float(avg_latency)


# TODO: Agregar más métricas acústicas aquí:
# def calculate_dynamic_range(forte_path: str, piano_path: str):
#     """Calcula el rango dinámico comparando forte vs piano."""
#     pass
#
# def calculate_spectral_centroid(audio_path: str):
#     """Calcula el centroide espectral (brillo del sonido)."""
#     pass


def extract_acoustic_features(audio_files: dict):
    """
    Extrae todas las características acústicas de los archivos proporcionados.

    Args:
        audio_files: Diccionario que mapea (category, technique) -> file_path
                     Ejemplo: {
                         ('long_tones', 'straight'): '/path/to/file1.wav',
                         ('long_tones', 'forte'): '/path/to/file2.wav',
                     }

    Returns:
        dict: Diccionario con todas las métricas acústicas calculadas
              Ejemplo: {'meanRmsDb': -25.79, 'dynamic_range': 15.3}
    """
    features = {}

    # Métrica 1: meanRmsDb (usa long_tones/straight)
    if ('long_tones', 'straight') in audio_files:
        features['meanRmsDb'] = calculate_mean_rms_db(
            audio_files[('long_tones', 'straight')]
        )

    # Métrica 2: rmsConsistency (usa long_tones/straight)
    if ('long_tones', 'straight') in audio_files:
        features['rmsConsistency'] = calculate_rms_consistency(
            audio_files[('long_tones', 'straight')]
        )

    # Métrica 3: dynamicRangeDb (usa long_tones/messa)
    if ('long_tones', 'messa') in audio_files:
        features['dynamicRangeDb'] = calculate_dynamic_range_db(
            audio_files[('long_tones', 'messa')]
        )

    # Métrica 4: durationSec (usa long_tones/straight)
    if ('long_tones', 'straight') in audio_files:
        features['durationSec'] = calculate_duration_sec(
            audio_files[('long_tones', 'straight')]
        )

    # Métrica 5: attackLatencyMs (usa long_tones/straight)
    if ('long_tones', 'straight') in audio_files:
        features['attackLatencyMs'] = calculate_attack_latency_ms(
            audio_files[('long_tones', 'straight')]
        )

    # TODO: Agregar más métricas acústicas aquí
    # Ejemplo de métrica que usa dos archivos:
    # if ('long_tones', 'forte') in audio_files and ('long_tones', 'pp') in audio_files:
    #     features['dynamic_range'] = calculate_dynamic_range(
    #         audio_files[('long_tones', 'forte')],
    #         audio_files[('long_tones', 'pp')]
    #     )

    return features


if __name__ == "__main__":
    """Test de las funciones acústicas."""
    from pathlib import Path
    from ursinger_ml.config import VOCALSET_ROOT

    root = Path(VOCALSET_ROOT)

    # Buscar archivos de ejemplo
    test_straight = None
    test_messa = None

    for singer_dir in root.iterdir():
        if singer_dir.is_dir():
            # Buscar straight
            if not test_straight:
                straight_path = singer_dir / "long_tones" / "straight"
                if straight_path.exists():
                    wav_files = list(straight_path.glob("*.wav"))
                    if wav_files:
                        test_straight = wav_files[0]

            # Buscar messa
            if not test_messa:
                messa_path = singer_dir / "long_tones" / "messa"
                if messa_path.exists():
                    wav_files = list(messa_path.glob("*.wav"))
                    if wav_files:
                        test_messa = wav_files[0]

            if test_straight and test_messa:
                break

    # Probar métricas individuales
    if test_straight:
        print(f"Probando con straight: {test_straight.name}")
        print(f"  meanRmsDb: {calculate_mean_rms_db(str(test_straight)):.2f} dBFS")
        print(f"  rmsConsistency: {calculate_rms_consistency(str(test_straight)):.2f} dB")
        print(f"  durationSec: {calculate_duration_sec(str(test_straight)):.2f} segundos")
        print(f"  attackLatencyMs: {calculate_attack_latency_ms(str(test_straight)):.2f} ms")

    if test_messa:
        print(f"\nProbando con messa: {test_messa.name}")
        print(f"  dynamicRangeDb: {calculate_dynamic_range_db(str(test_messa)):.2f} dB")

    # Probar extracción completa
    if test_straight or test_messa:
        print(f"\nExtraccion completa de caracteristicas acusticas:")
        audio_files = {}
        if test_straight:
            audio_files[('long_tones', 'straight')] = str(test_straight)
        if test_messa:
            audio_files[('long_tones', 'messa')] = str(test_messa)

        features = extract_acoustic_features(audio_files)
        for metric, value in features.items():
            print(f"  {metric}: {value:.2f}")
    else:
        print("No se encontraron archivos de prueba")

