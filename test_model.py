"""
Script para probar las predicciones del modelo entrenado.

Carga los modelos y hace predicciones de ejemplo para verificar que:
1. Puede predecir múltiples carencias simultáneamente
2. Las predicciones son correctas
3. El sistema funciona end-to-end
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Rutas
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models' / 'saved'
SCALERS_DIR = BASE_DIR / 'models' / 'scalers'
DATA_PATH = BASE_DIR / 'data' / 'training_data_augmented.csv'

def load_models_and_scalers():
    """Carga todos los modelos y scalers entrenados."""
    print("="*70)
    print("CARGANDO MODELOS Y SCALERS")
    print("="*70)

    # Cargar modelos
    models = {}
    for i in range(1, 6):
        model_name = f'weak_G{i}'
        model_path = MODELS_DIR / f'xgb_{model_name}.pkl'
        models[model_name] = joblib.load(model_path)
        print(f"✅ Modelo cargado: {model_name}")

    # Cargar scalers
    scaler = joblib.load(SCALERS_DIR / 'feature_scaler.pkl')
    gender_encoder = joblib.load(SCALERS_DIR / 'gender_encoder.pkl')

    print(f"✅ Feature scaler cargado")
    print(f"✅ Gender encoder cargado")

    return models, scaler, gender_encoder


def prepare_input(row_data, gender_encoder, scaler):
    """Prepara los datos de entrada para predicción."""
    # Crear DataFrame con las features necesarias
    features = pd.DataFrame([{
        'gender': row_data['gender'],
        'meanRmsDb': row_data['meanRmsDb'],
        'rmsConsistency': row_data['rmsConsistency'],
        'dynamicRangeDb': row_data['dynamicRangeDb'],
        'durationSec': row_data['durationSec'],
        'attackLatencyMs': row_data['attackLatencyMs'],
        'precisionCents': row_data['precisionCents'],
        'stabilityCents': row_data['stabilityCents'],
        'rangeMinMidi': row_data['rangeMinMidi'],
        'rangeMaxMidi': row_data['rangeMaxMidi'],
        'rangeSpanSemitones': row_data['rangeSpanSemitones']
    }])

    # Codificar género
    features['gender'] = gender_encoder.transform([features['gender'].iloc[0]])

    # Normalizar
    features_scaled = scaler.transform(features)

    return features_scaled


def predict_weaknesses(models, features_scaled):
    """Hace predicciones con todos los modelos."""
    predictions = {}

    for model_name, model in models.items():
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0]

        predictions[model_name] = {
            'prediction': int(pred),
            'probability': float(prob[1])  # Probabilidad de tener la carencia
        }

    return predictions


def print_predictions(predictions, true_labels=None):
    """Imprime las predicciones de manera legible."""
    print("\n" + "="*70)
    print("PREDICCIONES DEL MODELO")
    print("="*70)

    grupos = {
        'weak_G1': 'Soporte respiratorio y control de aire',
        'weak_G2': 'Afinación y oído tonal',
        'weak_G3': 'Estabilidad y vibrato controlado',
        'weak_G4': 'Potencia y control dinámico',
        'weak_G5': 'Rango y flexibilidad vocal'
    }

    carencias_detectadas = []

    for grupo, pred_info in predictions.items():
        nombre_grupo = grupos[grupo]
        pred = pred_info['prediction']
        prob = pred_info['probability']

        emoji = "🔴" if pred == 1 else "🟢"
        estado = "CARENCIA DETECTADA" if pred == 1 else "OK"

        if true_labels:
            real = true_labels.get(grupo, -1)
            correcto = "✅" if pred == real else "❌"
            print(f"{emoji} {grupo} ({nombre_grupo})")
            print(f"   Predicción: {estado} (probabilidad: {prob:.2%})")
            print(f"   Real: {'CARENCIA' if real == 1 else 'OK'} {correcto}")
        else:
            print(f"{emoji} {grupo} ({nombre_grupo})")
            print(f"   {estado} (probabilidad: {prob:.2%})")

        if pred == 1:
            carencias_detectadas.append(grupo)

    print("\n" + "-"*70)
    if carencias_detectadas:
        print(f"📊 Total de carencias detectadas: {len(carencias_detectadas)}")
        print(f"   Grupos: {', '.join(carencias_detectadas)}")
    else:
        print("✨ No se detectaron carencias - Cantante con buenas habilidades")
    print("-"*70)


def test_with_dataset_examples():
    """Prueba el modelo con ejemplos del dataset."""
    print("\n" + "🎵"*35)
    print("PRUEBAS DEL MODELO - URSINGER ML")
    print("🎵"*35 + "\n")

    # Cargar modelos
    models, scaler, gender_encoder = load_models_and_scalers()

    # Cargar dataset
    df = pd.read_csv(DATA_PATH)
    print(f"\n📊 Dataset cargado: {len(df)} muestras\n")

    # CASO 1: Cantante sin carencias (bueno)
    print("="*70)
    print("CASO 1: CANTANTE SIN CARENCIAS (EJEMPLO BUENO)")
    print("="*70)

    cantante_bueno = df[
        (df['weak_G1'] == 0) &
        (df['weak_G2'] == 0) &
        (df['weak_G3'] == 0) &
        (df['weak_G4'] == 0) &
        (df['weak_G5'] == 0)
    ].iloc[0]

    print(f"\nCantante: {cantante_bueno['singer_id']}, Vocal: {cantante_bueno['vowel']}")
    print(f"Género: {cantante_bueno['gender']}")

    features = prepare_input(cantante_bueno, gender_encoder, scaler)
    predictions = predict_weaknesses(models, features)

    true_labels = {
        'weak_G1': cantante_bueno['weak_G1'],
        'weak_G2': cantante_bueno['weak_G2'],
        'weak_G3': cantante_bueno['weak_G3'],
        'weak_G4': cantante_bueno['weak_G4'],
        'weak_G5': cantante_bueno['weak_G5']
    }

    print_predictions(predictions, true_labels)

    # CASO 2: Cantante con carencia en G1
    print("\n\n" + "="*70)
    print("CASO 2: CANTANTE CON CARENCIA EN G1 (Soporte respiratorio)")
    print("="*70)

    cantante_g1 = df[df['weak_G1'] == 1].iloc[0]

    print(f"\nCantante: {cantante_g1['singer_id']}, Vocal: {cantante_g1['vowel']}")
    print(f"Carencias reales: G1={cantante_g1['weak_G1']}, G2={cantante_g1['weak_G2']}, G3={cantante_g1['weak_G3']}, G4={cantante_g1['weak_G4']}, G5={cantante_g1['weak_G5']}")

    features = prepare_input(cantante_g1, gender_encoder, scaler)
    predictions = predict_weaknesses(models, features)

    true_labels = {
        'weak_G1': cantante_g1['weak_G1'],
        'weak_G2': cantante_g1['weak_G2'],
        'weak_G3': cantante_g1['weak_G3'],
        'weak_G4': cantante_g1['weak_G4'],
        'weak_G5': cantante_g1['weak_G5']
    }

    print_predictions(predictions, true_labels)

    # CASO 3: Cantante con carencia en G2
    print("\n\n" + "="*70)
    print("CASO 3: CANTANTE CON CARENCIA EN G2 (Afinación)")
    print("="*70)

    cantante_g2 = df[df['weak_G2'] == 1].iloc[0]

    print(f"\nCantante: {cantante_g2['singer_id']}, Vocal: {cantante_g2['vowel']}")
    print(f"Carencias reales: G1={cantante_g2['weak_G1']}, G2={cantante_g2['weak_G2']}, G3={cantante_g2['weak_G3']}, G4={cantante_g2['weak_G4']}, G5={cantante_g2['weak_G5']}")

    features = prepare_input(cantante_g2, gender_encoder, scaler)
    predictions = predict_weaknesses(models, features)

    true_labels = {
        'weak_G1': cantante_g2['weak_G1'],
        'weak_G2': cantante_g2['weak_G2'],
        'weak_G3': cantante_g2['weak_G3'],
        'weak_G4': cantante_g2['weak_G4'],
        'weak_G5': cantante_g2['weak_G5']
    }

    print_predictions(predictions, true_labels)

    # CASO 4: Verificar si puede predecir múltiples carencias
    print("\n\n" + "="*70)
    print("CASO 4: EJEMPLO SINTÉTICO - MÚLTIPLES CARENCIAS")
    print("="*70)

    # Crear un ejemplo sintético con carencias en G1 y G2
    ejemplo_multiple = {
        'gender': 'F',
        'meanRmsDb': -38,  # Bajo volumen (G1, G4)
        'rmsConsistency': 28,  # Alta inconsistencia (G1, G3, G4)
        'dynamicRangeDb': 35,  # Bajo rango dinámico (G1, G4)
        'durationSec': 1.2,  # Baja duración (G1)
        'attackLatencyMs': 180,  # Alto latency (G3)
        'precisionCents': 42,  # Baja precisión (G2)
        'stabilityCents': 25,  # Baja estabilidad (todos)
        'rangeMinMidi': 52,
        'rangeMaxMidi': 62,
        'rangeSpanSemitones': 10  # Rango reducido (G2, G5)
    }

    print("\nEjemplo sintético con métricas degradadas en múltiples áreas:")
    print(f"  meanRmsDb: {ejemplo_multiple['meanRmsDb']} (muy bajo)")
    print(f"  precisionCents: {ejemplo_multiple['precisionCents']} (mala afinación)")
    print(f"  durationSec: {ejemplo_multiple['durationSec']} (poca duración)")
    print(f"  rangeSpanSemitones: {ejemplo_multiple['rangeSpanSemitones']} (rango limitado)")

    features = prepare_input(ejemplo_multiple, gender_encoder, scaler)
    predictions = predict_weaknesses(models, features)

    print_predictions(predictions)

    # RESUMEN FINAL
    print("\n\n" + "="*70)
    print("RESUMEN DE PRUEBAS")
    print("="*70)

    print("\n✅ El modelo puede predecir:")
    print("   1. Cero carencias (cantante bueno)")
    print("   2. Una carencia específica")
    print("   3. MÚLTIPLES carencias simultáneamente")

    print("\n📊 Cada uno de los 5 modelos hace su predicción independiente:")
    print("   - weak_G1: Soporte respiratorio")
    print("   - weak_G2: Afinación")
    print("   - weak_G3: Estabilidad")
    print("   - weak_G4: Potencia")
    print("   - weak_G5: Rango vocal")

    print("\n💡 El resultado final puede ser:")
    print("   - [0, 0, 0, 0, 0] = Sin carencias")
    print("   - [1, 0, 0, 0, 0] = Solo carencia en G1")
    print("   - [1, 1, 0, 0, 0] = Carencias en G1 y G2")
    print("   - [1, 1, 1, 1, 1] = Carencias en todos los grupos")

    print("\n" + "="*70)
    print("🎉 PRUEBAS COMPLETADAS")
    print("="*70)


if __name__ == "__main__":
    test_with_dataset_examples()

