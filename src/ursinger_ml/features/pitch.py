"""
Métricas relacionadas con el pitch (altura tonal).

Este módulo contendrá funciones para calcular características del pitch
como vibrato, estabilidad tonal, rango vocal, etc.
"""
import numpy as np
import librosa


def hz_to_midi(frequency_hz):
    """
    Convierte frecuencia en Hz a nota MIDI.

    Args:
        frequency_hz: Frecuencia en Hz

    Returns:
        float: Nota MIDI (ej: 69.0 = A4)
    """
    return 12 * np.log2(frequency_hz / 440.0) + 69


def midi_to_hz(midi_note):
    """
    Convierte nota MIDI a frecuencia en Hz.

    Args:
        midi_note: Nota MIDI (ej: 69 = A4)

    Returns:
        float: Frecuencia en Hz
    """
    return 440.0 * (2 ** ((midi_note - 69) / 12))


def cents_difference(freq_hz, target_midi):
    """
    Calcula la diferencia en cents entre una frecuencia y una nota objetivo.

    Args:
        freq_hz: Frecuencia emitida en Hz
        target_midi: Nota MIDI objetivo

    Returns:
        float: Diferencia en cents (100 cents = 1 semitono)
    """
    target_hz = midi_to_hz(target_midi)
    cents = 1200 * np.log2(freq_hz / target_hz)
    return cents


def calculate_precision_cents(audio_path: str, sr: int = 22050):
    """
    Calcula la precisión promedio de afinación en cents.

    Usa archivos de escalas (scales/straight) donde el cantante canta
    una secuencia de notas. Para cada nota:
    1. Detecta el pitch con CREPE
    2. Segmenta por cambios bruscos de pitch
    3. Determina la nota objetivo (mediana → MIDI redondeado)
    4. Calcula el error en cents
    5. Promedia los errores absolutos

    Args:
        audio_path: Ruta al archivo de audio .wav (scales/straight)
        sr: Sample rate (por defecto 22050 Hz)

    Returns:
        float: Error promedio de afinación en cents

    Interpretación:
        - 0-10 cents: Afinación excelente (profesional)
        - 10-25 cents: Buena afinación (avanzado)
        - 25-50 cents: Afinación moderada (intermedio)
        - >50 cents: Necesita trabajo (principiante)

    Ejemplo:
        >>> precision = calculate_precision_cents('scale_a.wav')
        >>> print(f"Precisión: {precision:.1f} cents")
        Precisión: 15.3 cents
    """
    import crepe

    # Cargar el audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Usar CREPE para detectar pitch
    time, frequency, confidence, activation = crepe.predict(
        y, sr,
        viterbi=True,
        step_size=50,  # 50ms para velocidad
        verbose=0,
        model_capacity='tiny'
    )

    # Filtrar por confianza
    conf_threshold = 0.6  # Umbral más alto para mayor precisión
    valid_mask = confidence > conf_threshold

    if np.sum(valid_mask) < 3:
        # No hay suficientes frames válidos
        return 0.0

    frequency_valid = frequency[valid_mask]
    time_valid = time[valid_mask]

    # Convertir a MIDI
    midi_values = hz_to_midi(frequency_valid)

    # Detectar cambios de nota (cambios bruscos de pitch)
    midi_diff = np.abs(np.diff(midi_values))
    note_changes = midi_diff > 0.5  # Más de medio semitono = cambio de nota

    # Índices de cambio
    change_indices = np.concatenate([[0], np.where(note_changes)[0] + 1, [len(midi_values)]])

    # Calcular error por segmento (por nota)
    segment_errors = []

    for i in range(len(change_indices) - 1):
        start_idx = change_indices[i]
        end_idx = change_indices[i + 1]

        segment_length = end_idx - start_idx

        # Solo procesar segmentos con al menos 3 frames
        if segment_length < 3:
            continue

        # Excluir frames de transición (primeros y últimos 20% del segmento)
        trim_size = max(1, int(segment_length * 0.2))
        start_trim = start_idx + trim_size
        end_trim = end_idx - trim_size

        if start_trim >= end_trim:
            # Segmento muy corto, usar todo
            start_trim = start_idx
            end_trim = end_idx

        segment_midi = midi_values[start_trim:end_trim]
        segment_freq = frequency_valid[start_trim:end_trim]

        if len(segment_midi) == 0:
            continue

        # Nota objetivo: mediana de MIDI redondeada al entero más cercano
        target_midi = np.round(np.median(segment_midi))

        # Calcular error en cents para cada frame del segmento
        errors_cents = cents_difference(segment_freq, target_midi)

        # Error absoluto promedio del segmento
        mae_segment = np.mean(np.abs(errors_cents))
        segment_errors.append(mae_segment)

    # Promedio de errores de todos los segmentos
    if segment_errors:
        avg_error = np.mean(segment_errors)
    else:
        avg_error = 0.0

    return float(avg_error)


def calculate_stability_cents(audio_path: str, sr: int = 22050):
    """
    Calcula la estabilidad del pitch durante notas sostenidas.

    Usa archivos de long_tones/straight donde el cantante sostiene 3 notas.
    Para cada nota sostenida:
    1. Detecta el pitch con CREPE
    2. Segmenta por cambios bruscos de pitch
    3. Calcula la desviación estándar del pitch en cents
    4. Promedia las desviaciones de todas las notas

    Args:
        audio_path: Ruta al archivo de audio .wav (long_tones/straight)
        sr: Sample rate (por defecto 22050 Hz)

    Returns:
        float: Desviación estándar promedio en cents

    Interpretación:
        - 0-5 cents: Estabilidad excelente (vibrato controlado o tono muy estable)
        - 5-15 cents: Buena estabilidad (control profesional)
        - 15-30 cents: Estabilidad moderada (vibrato natural o fluctuaciones leves)
        - >30 cents: Baja estabilidad (fluctuaciones notables)

    Ejemplo:
        >>> stability = calculate_stability_cents('long_straight_a.wav')
        >>> print(f"Estabilidad: {stability:.1f} cents")
        Estabilidad: 12.5 cents
    """
    import crepe

    # Cargar el audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Usar CREPE para detectar pitch
    time, frequency, confidence, activation = crepe.predict(
        y, sr,
        viterbi=True,
        step_size=50,  # 50ms para velocidad
        verbose=0,
        model_capacity='tiny'
    )

    # Filtrar por confianza
    conf_threshold = 0.6
    valid_mask = confidence > conf_threshold

    if np.sum(valid_mask) < 3:
        return 0.0

    frequency_valid = frequency[valid_mask]
    time_valid = time[valid_mask]

    # Convertir a MIDI
    midi_values = hz_to_midi(frequency_valid)

    # Detectar cambios de nota (segmentar notas sostenidas)
    midi_diff = np.abs(np.diff(midi_values))
    note_changes = midi_diff > 0.5  # Más de medio semitono = cambio de nota

    # Índices de cambio
    change_indices = np.concatenate([[0], np.where(note_changes)[0] + 1, [len(midi_values)]])

    # Calcular desviación estándar por segmento
    segment_stds = []

    for i in range(len(change_indices) - 1):
        start_idx = change_indices[i]
        end_idx = change_indices[i + 1]

        segment_length = end_idx - start_idx

        # Solo procesar segmentos con al menos 5 frames
        if segment_length < 5:
            continue

        # Excluir frames de transición (primeros y últimos 20%)
        trim_size = max(1, int(segment_length * 0.2))
        start_trim = start_idx + trim_size
        end_trim = end_idx - trim_size

        if start_trim >= end_trim:
            start_trim = start_idx
            end_trim = end_idx

        segment_midi = midi_values[start_trim:end_trim]

        if len(segment_midi) < 3:
            continue

        # Nota objetivo: mediana redondeada
        target_midi = np.round(np.median(segment_midi))

        # Convertir diferencias a cents
        # cents = diferencia en semitonos * 100
        cents_deviations = (segment_midi - target_midi) * 100

        # Desviación estándar en cents
        std_cents = np.std(cents_deviations)
        segment_stds.append(std_cents)

    # Promedio de desviaciones estándar
    if segment_stds:
        avg_stability = np.mean(segment_stds)
    else:
        avg_stability = 0.0

    return float(avg_stability)


def calculate_range_metrics(audio_path_c: str = None, audio_path_f: str = None, sr: int = 22050):
    """
    Calcula el rango vocal usando archivos de escalas _c_ (bajo) y _f_ (alto).

    Args:
        audio_path_c: Archivo con escala baja (_c_fast_piano) o None
        audio_path_f: Archivo con escala alta (_f_fast_piano) o None
        sr: Sample rate

    Returns:
        dict: {'rangeMinMidi': nota_min, 'rangeMaxMidi': nota_max, 'rangeSpanSemitones': span}

    Descripción:
        - Si ambos archivos disponibles: usa _c_ para min, _f_ para max
        - Si solo _c_ disponible: usa _c_ para min y max
        - Si solo _f_ disponible: usa _f_ para min y max
        - rangeSpanSemitones: Diferencia entre min y max

    Interpretación de rangeSpanSemitones:
        - 12-18 semitonos (1-1.5 octavas): Rango básico
        - 18-24 semitonos (1.5-2 octavas): Rango bueno
        - 24-36 semitonos (2-3 octavas): Rango amplio
        - >36 semitonos (>3 octavas): Rango excepcional
    """
    import crepe

    # Determinar qué archivos usar
    if audio_path_c and audio_path_f:
        # Caso ideal: ambos archivos disponibles
        use_c_for_min = audio_path_c
        use_f_for_max = audio_path_f
    elif audio_path_c:
        # Solo _c_ disponible: usar para ambos
        use_c_for_min = audio_path_c
        use_f_for_max = audio_path_c
    elif audio_path_f:
        # Solo _f_ disponible: usar para ambos
        use_c_for_min = audio_path_f
        use_f_for_max = audio_path_f
    else:
        # No hay archivos disponibles
        return {
            'rangeMinMidi': 0.0,
            'rangeMaxMidi': 0.0,
            'rangeSpanSemitones': 0.0
        }

    # Procesar archivo para mínimo
    y_c, sr = librosa.load(use_c_for_min, sr=sr)
    time_c, freq_c, conf_c, _ = crepe.predict(
        y_c, sr,
        viterbi=True,
        step_size=50,
        verbose=0,
        model_capacity='tiny'
    )

    # Procesar archivo para máximo (puede ser el mismo)
    if use_f_for_max == use_c_for_min:
        # Reutilizar resultados
        freq_f = freq_c
        conf_f = conf_c
    else:
        y_f, sr = librosa.load(use_f_for_max, sr=sr)
        time_f, freq_f, conf_f, _ = crepe.predict(
            y_f, sr,
            viterbi=True,
            step_size=50,
            verbose=0,
            model_capacity='tiny'
        )

    # Filtrar por confianza
    conf_threshold = 0.6
    valid_c = conf_c > conf_threshold
    valid_f = conf_f > conf_threshold

    # Convertir a MIDI
    if np.sum(valid_c) > 0:
        midi_c = hz_to_midi(freq_c[valid_c])
        range_min = np.min(midi_c)
    else:
        range_min = 0.0

    if np.sum(valid_f) > 0:
        midi_f = hz_to_midi(freq_f[valid_f])
        range_max = np.max(midi_f)
    else:
        range_max = 0.0

    # Calcular span (diferencia)
    if range_min > 0 and range_max > 0:
        range_span = range_max - range_min
    else:
        range_span = 0.0

    return {
        'rangeMinMidi': float(range_min),
        'rangeMaxMidi': float(range_max),
        'rangeSpanSemitones': float(range_span)
    }


def calculate_vibrato_metrics(audio_path: str, sr: int = 22050):
    """
    Calcula métricas de vibrato: frecuencia (rate) y amplitud (depth).

    Args:
        audio_path: Ruta al archivo de audio .wav (long_tones/forte)
        sr: Sample rate

    Returns:
        dict: {'vibratoRateHz': frecuencia, 'vibratoDepthCents': amplitud}

    Descripción:
        En archivos de long_tones/forte, el cantante sostiene 3 notas con vibrato.
        Para cada nota:
        1. Detecta el pitch con CREPE
        2. Segmenta por cambios bruscos de pitch
        3. Calcula la frecuencia de oscilación (rate en Hz)
        4. Calcula la amplitud de oscilación (depth en cents)
        5. Promedia los valores de los 3 segmentos

    Interpretación de vibratoRateHz:
        - 4-5 Hz: Vibrato lento (barroco/operístico)
        - 5-6 Hz: Vibrato normal (típico)
        - 6-7 Hz: Vibrato rápido
        - >7 Hz: Vibrato muy rápido (puede sonar nervioso)

    Interpretación de vibratoDepthCents:
        - 20-40 cents: Vibrato sutil (pop/jazz)
        - 40-80 cents: Vibrato moderado (clásico)
        - 80-120 cents: Vibrato amplio (operístico)
        - >120 cents: Vibrato muy amplio

    Ejemplo:
        >>> vibrato = calculate_vibrato_metrics('forte_a.wav')
        >>> print(f"Rate: {vibrato['vibratoRateHz']:.2f} Hz")
        >>> print(f"Depth: {vibrato['vibratoDepthCents']:.2f} cents")
        Rate: 5.80 Hz
        Depth: 65.40 cents
    """
    import crepe
    from scipy import signal

    # Cargar el audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Usar CREPE para detectar pitch
    time, frequency, confidence, _ = crepe.predict(
        y, sr,
        viterbi=True,
        step_size=10,  # Menor step_size para mejor resolución temporal del vibrato
        verbose=0,
        model_capacity='tiny'
    )

    # Filtrar por confianza
    conf_threshold = 0.6
    valid_mask = confidence > conf_threshold

    if np.sum(valid_mask) < 10:
        return {'vibratoRateHz': 0.0, 'vibratoDepthCents': 0.0}

    frequency_valid = frequency[valid_mask]
    time_valid = time[valid_mask]

    # Convertir a MIDI para facilitar la segmentación
    midi_values = hz_to_midi(frequency_valid)

    # Detectar cambios de nota (segmentar las 3 notas)
    midi_diff = np.abs(np.diff(midi_values))
    note_changes = midi_diff > 0.8  # Umbral más alto para evitar falsos positivos

    # Índices de cambio
    change_indices = np.concatenate([[0], np.where(note_changes)[0] + 1, [len(midi_values)]])

    # Calcular vibrato por segmento
    segment_rates = []
    segment_depths = []

    for i in range(len(change_indices) - 1):
        start_idx = change_indices[i]
        end_idx = change_indices[i + 1]

        segment_length = end_idx - start_idx

        # Solo procesar segmentos con al menos 15 frames
        if segment_length < 15:
            continue

        # Excluir transiciones (primeros y últimos 15%)
        trim_size = max(1, int(segment_length * 0.15))
        start_trim = start_idx + trim_size
        end_trim = end_idx - trim_size

        if start_trim >= end_trim or (end_trim - start_trim) < 8:
            continue

        segment_freq = frequency_valid[start_trim:end_trim]
        segment_time = time_valid[start_trim:end_trim]
        segment_midi = midi_values[start_trim:end_trim]

        if len(segment_freq) < 8:
            continue

        # Calcular time_diff una vez para todo el segmento
        time_diff = np.mean(np.diff(segment_time)) if len(segment_time) > 1 else 0.01

        # Detrend el pitch para análisis de vibrato
        midi_detrended = signal.detrend(segment_midi)

        # Calcular vibrato rate (frecuencia de oscilación)
        # Método 1: Autocorrelación
        autocorr = np.correlate(midi_detrended, midi_detrended, mode='full')
        autocorr = autocorr[len(autocorr)//2:]

        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        from scipy.signal import find_peaks
        peaks, properties = find_peaks(autocorr[1:], height=0.2, distance=2)

        rate_found = False
        if len(peaks) > 0:
            period_frames = peaks[0] + 1
            period_sec = period_frames * time_diff

            if period_sec > 0:
                rate_hz = 1.0 / period_sec
                if 2.5 <= rate_hz <= 12.0:
                    segment_rates.append(rate_hz)
                    rate_found = True

        # Método 2: FFT (si autocorrelación no funcionó)
        if not rate_found and len(midi_detrended) > 8:
            try:
                # Calcular FFT
                fft_vals = np.fft.rfft(midi_detrended)
                fft_freq = np.fft.rfftfreq(len(midi_detrended), d=time_diff)

                # Magnitud del espectro
                fft_mag = np.abs(fft_vals)

                # Buscar el pico dominante (excluyendo DC y muy bajas frecuencias)
                valid_freq_mask = (fft_freq > 2.0) & (fft_freq < 12.0)
                if np.any(valid_freq_mask):
                    valid_fft_mag = fft_mag[valid_freq_mask]
                    valid_fft_freq = fft_freq[valid_freq_mask]

                    if len(valid_fft_mag) > 0:
                        peak_idx = np.argmax(valid_fft_mag)
                        rate_hz = valid_fft_freq[peak_idx]

                        if 2.5 <= rate_hz <= 12.0:
                            segment_rates.append(rate_hz)
            except Exception:
                # Si FFT falla, continuar sin agregar rate
                pass

        # Calcular vibrato depth (amplitud de oscilación)
        midi_peak_to_peak = np.max(midi_detrended) - np.min(midi_detrended)
        depth_cents = midi_peak_to_peak * 100

        # Filtrar valores poco realistas - rango muy amplio para capturar vibrato sutil
        if 2.0 <= depth_cents <= 300.0:
            segment_depths.append(depth_cents)

    # Promediar resultados de todos los segmentos
    avg_rate = np.mean(segment_rates) if segment_rates else 0.0
    avg_depth = np.mean(segment_depths) if segment_depths else 0.0

    # Si no hay depth pero hay segmentos procesados, calcular depth de manera más simple
    # Esto maneja casos donde el vibrato es muy sutil
    if avg_depth == 0.0 and len(midi_values) > 10:
        # Usar toda la señal detrended
        full_detrended = signal.detrend(midi_values)
        full_peak_to_peak = np.max(full_detrended) - np.min(full_detrended)
        full_depth = full_peak_to_peak * 100

        # Aceptar valores más bajos (incluso vibrato muy sutil)
        if full_depth >= 2.0:  # Umbral muy bajo
            avg_depth = full_depth

    return {
        'vibratoRateHz': float(avg_rate),
        'vibratoDepthCents': float(avg_depth)
    }


# TODO: Implementar más métricas de pitch

# def calculate_vibrato_rate(audio_path: str, sr: int = 22050):
#     """
#     Calcula la velocidad de vibrato en Hz.
#
#     Args:
#         audio_path: Ruta al archivo de audio .wav
#         sr: Sample rate
#
#     Returns:
#         float: Velocidad de vibrato en Hz
#     """
#     import crepe
#
#     y, sr = librosa.load(audio_path, sr=sr)
#
#     # Usar CREPE para detectar pitch
#     time, frequency, confidence, activation = crepe.predict(y, sr, viterbi=True)
#
#     # Filtrar por confianza
#     frequency = frequency[confidence > 0.5]
#
#     # Calcular la frecuencia del vibrato (modulación del pitch)
#     # TODO: Implementar detección de vibrato
#
#     return vibrato_rate


# def calculate_pitch_stability(audio_path: str):
#     """Calcula la estabilidad del pitch (desviación estándar)."""
#     pass


# def calculate_vocal_range(audio_paths: list):
#     """Calcula el rango vocal en semitonos."""
#     pass


def extract_pitch_features(audio_files: dict):
    """
    Extrae todas las características de pitch de los archivos proporcionados.

    Args:
        audio_files: Diccionario que mapea (category, technique, pitch_type) -> file_path
                     Para métricas de rango, debe incluir:
                     - ('scales', 'fast_piano', 'c') para nota mínima
                     - ('scales', 'fast_piano', 'f') para nota máxima

    Returns:
        dict: Diccionario con todas las métricas de pitch calculadas
    """
    features = {}

    # Métrica: precisionCents (usa scales/straight)
    if ('scales', 'straight') in audio_files:
        features['precisionCents'] = calculate_precision_cents(
            audio_files[('scales', 'straight')]
        )

    # Métrica: stabilityCents (usa long_tones/straight)
    if ('long_tones', 'straight') in audio_files:
        features['stabilityCents'] = calculate_stability_cents(
            audio_files[('long_tones', 'straight')]
        )

    # Métricas de rango: requiere al menos uno de _c_ o _f_
    key_c = ('scales', 'fast_piano', 'c')
    key_f = ('scales', 'fast_piano', 'f')

    has_c = key_c in audio_files
    has_f = key_f in audio_files

    if has_c or has_f:
        audio_c = audio_files.get(key_c)
        audio_f = audio_files.get(key_f)

        range_metrics = calculate_range_metrics(audio_c, audio_f)
        features.update(range_metrics)

    # Métricas de vibrato: vibratoRateHz y vibratoDepthCents (usa long_tones/forte)
    if ('long_tones', 'forte') in audio_files:
        vibrato_metrics = calculate_vibrato_metrics(
            audio_files[('long_tones', 'forte')]
        )
        features.update(vibrato_metrics)

    # TODO: Implementar más métricas de pitch

    return features


if __name__ == "__main__":
    """Test de las funciones de pitch."""
    from pathlib import Path
    from ursinger_ml.config import VOCALSET_ROOT

    root = Path(VOCALSET_ROOT)

    # Buscar archivos de ejemplo
    test_scale = None
    test_long_tone = None
    test_scale_c = None
    test_scale_f = None
    test_forte = None

    for singer_dir in root.iterdir():
        if not singer_dir.is_dir():
            continue

        # Buscar scales/straight
        if not test_scale:
            scales_path = singer_dir / "scales" / "straight"
            if scales_path.exists():
                wav_files = list(scales_path.glob("*.wav"))
                if wav_files:
                    test_scale = wav_files[0]

        # Buscar long_tones/straight
        if not test_long_tone:
            long_tones_path = singer_dir / "long_tones" / "straight"
            if long_tones_path.exists():
                wav_files = list(long_tones_path.glob("*.wav"))
                if wav_files:
                    test_long_tone = wav_files[0]

        # Buscar long_tones/forte
        if not test_forte:
            forte_path = singer_dir / "long_tones" / "forte"
            if forte_path.exists():
                wav_files = list(forte_path.glob("*.wav"))
                if wav_files:
                    test_forte = wav_files[0]

        # Buscar scales/fast_piano con _c_ y _f_
        if not test_scale_c or not test_scale_f:
            fast_piano_path = singer_dir / "scales" / "fast_piano"
            if fast_piano_path.exists():
                for wav in fast_piano_path.glob("*.wav"):
                    if '_c_' in wav.name.lower() and not test_scale_c:
                        test_scale_c = wav
                    elif '_f_' in wav.name.lower() and not test_scale_f:
                        test_scale_f = wav

        if test_scale and test_long_tone and test_scale_c and test_scale_f and test_forte:
            break

    # Probar métricas individuales
    if test_scale:
        print(f"Probando precisionCents con scales/straight: {test_scale.name}")
        precision = calculate_precision_cents(str(test_scale))
        print(f"  precisionCents: {precision:.2f} cents")

    if test_long_tone:
        print(f"\nProbando stabilityCents con long_tones/straight: {test_long_tone.name}")
        stability = calculate_stability_cents(str(test_long_tone))
        print(f"  stabilityCents: {stability:.2f} cents")

    if test_scale_c and test_scale_f:
        print(f"\nProbando rangeMetrics con scales/fast_piano:")
        print(f"  Archivo _c_: {test_scale_c.name}")
        print(f"  Archivo _f_: {test_scale_f.name}")
        range_metrics = calculate_range_metrics(str(test_scale_c), str(test_scale_f))
        for metric, value in range_metrics.items():
            print(f"  {metric}: {value:.2f}")
    elif test_scale_c:
        print(f"\nProbando rangeMetrics con scales/fast_piano (solo _c_):")
        print(f"  Archivo _c_: {test_scale_c.name}")
        range_metrics = calculate_range_metrics(audio_path_c=str(test_scale_c))
        for metric, value in range_metrics.items():
            print(f"  {metric}: {value:.2f}")
    elif test_scale_f:
        print(f"\nProbando rangeMetrics con scales/fast_piano (solo _f_):")
        print(f"  Archivo _f_: {test_scale_f.name}")
        range_metrics = calculate_range_metrics(audio_path_f=str(test_scale_f))
        for metric, value in range_metrics.items():
            print(f"  {metric}: {value:.2f}")

    if test_forte:
        print(f"\nProbando vibratoMetrics con long_tones/forte: {test_forte.name}")
        vibrato_metrics = calculate_vibrato_metrics(str(test_forte))
        for metric, value in vibrato_metrics.items():
            print(f"  {metric}: {value:.2f}")

    # Probar extracción completa
    print(f"\nExtraccion completa de caracteristicas de pitch:")
    audio_files = {}
    if test_scale:
        audio_files[('scales', 'straight')] = str(test_scale)
    if test_long_tone:
        audio_files[('long_tones', 'straight')] = str(test_long_tone)
    if test_scale_c and test_scale_f:
        audio_files[('scales', 'fast_piano', 'c')] = str(test_scale_c)
        audio_files[('scales', 'fast_piano', 'f')] = str(test_scale_f)
    if test_forte:
        audio_files[('long_tones', 'forte')] = str(test_forte)

    features = extract_pitch_features(audio_files)
    for metric, value in features.items():
        print(f"  {metric}: {value:.2f}")

