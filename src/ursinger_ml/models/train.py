"""
Entrenamiento del modelo XGBoost para clasificación de debilidades vocales.

Este módulo entrena modelos multi-output para predecir debilidades en 5 grupos de habilidad:
- G1: Soporte respiratorio y control de aire
- G2: Afinación y oído tonal
- G3: Estabilidad y vibrato controlado
- G4: Potencia y control dinámico
- G5: Rango y flexibilidad vocal
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import xgboost as xgb
import joblib
import json

# Configuración
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Rutas
DATA_PATH = Path(__file__).parent.parent.parent.parent / 'data' / 'training_data_augmented.csv'
MODELS_DIR = Path(__file__).parent.parent.parent.parent / 'models' / 'saved'
SCALERS_DIR = Path(__file__).parent.parent.parent.parent / 'models' / 'scalers'
REPORTS_DIR = Path(__file__).parent.parent.parent.parent / 'models' / 'reports'

# Crear directorios si no existen
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCALERS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Carga el dataset aumentado."""
    print("="*70)
    print("CARGANDO DATOS")
    print("="*70)

    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset cargado: {df.shape}")
    print(f"Columnas: {list(df.columns)}")

    return df


def prepare_features_and_targets(df):
    """Separa features (X) y targets (y)."""
    print("\n" + "="*70)
    print("PREPARANDO FEATURES Y TARGETS")
    print("="*70)

    # Features (X)
    feature_columns = [
        'gender',
        'meanRmsDb',
        'rmsConsistency',
        'dynamicRangeDb',
        'durationSec',
        'attackLatencyMs',
        'precisionCents',
        'stabilityCents',
        'rangeMinMidi',
        'rangeMaxMidi',
        'rangeSpanSemitones'
    ]

    # Targets (y)
    target_columns = [
        'weak_G1',
        'weak_G2',
        'weak_G3',
        'weak_G4',
        'weak_G5'
    ]

    print(f"\nFeatures seleccionados ({len(feature_columns)}):")
    for feat in feature_columns:
        print(f"  - {feat}")

    print(f"\nTargets ({len(target_columns)}):")
    for target in target_columns:
        count = df[target].sum()
        print(f"  - {target}: {count} positivos ({count/len(df)*100:.1f}%)")

    X = df[feature_columns].copy()
    y = df[target_columns].copy()

    # Codificar género (F=0, M=1)
    le = LabelEncoder()
    X['gender'] = le.fit_transform(X['gender'])

    # Guardar el encoder
    joblib.dump(le, SCALERS_DIR / 'gender_encoder.pkl')
    print(f"\n✅ Gender encoder guardado en: {SCALERS_DIR / 'gender_encoder.pkl'}")

    return X, y, feature_columns, target_columns


def split_data(X, y):
    """Divide datos en train/test."""
    print("\n" + "="*70)
    print("DIVIDIENDO DATOS (TRAIN/TEST)")
    print("="*70)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y['weak_G1']  # Estratificar por G1 para balance
    )

    print(f"\nTrain set: {X_train.shape[0]} muestras ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"Test set: {X_test.shape[0]} muestras ({TEST_SIZE*100:.0f}%)")

    # Mostrar distribución
    print("\nDistribución en Train:")
    for col in y_train.columns:
        count = y_train[col].sum()
        print(f"  {col}: {count} positivos ({count/len(y_train)*100:.1f}%)")

    print("\nDistribución en Test:")
    for col in y_test.columns:
        count = y_test[col].sum()
        print(f"  {col}: {count} positivos ({count/len(y_test)*100:.1f}%)")

    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test):
    """Normaliza las features usando StandardScaler."""
    print("\n" + "="*70)
    print("NORMALIZANDO FEATURES")
    print("="*70)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convertir de nuevo a DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Guardar scaler
    scaler_path = SCALERS_DIR / 'feature_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"\n✅ Scaler guardado en: {scaler_path}")

    print("\nEjemplo de normalización (primeras 3 features, primera muestra):")
    print(f"  Original: {X_train.iloc[0, :3].values}")
    print(f"  Normalizado: {X_train_scaled.iloc[0, :3].values}")

    return X_train_scaled, X_test_scaled, scaler


def train_models(X_train, y_train, target_columns):
    """Entrena un modelo XGBoost para cada target."""
    print("\n" + "="*70)
    print("ENTRENANDO MODELOS XGBOOST")
    print("="*70)

    models = {}

    # Parámetros de XGBoost
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': RANDOM_STATE,
        'use_label_encoder': False
    }

    print(f"\nParámetros de XGBoost:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    for target in target_columns:
        print(f"\n📊 Entrenando modelo para {target}...")

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train[target])

        models[target] = model

        # Guardar modelo
        model_path = MODELS_DIR / f'xgb_{target}.pkl'
        joblib.dump(model, model_path)
        print(f"   ✅ Modelo guardado en: {model_path}")

    print(f"\n✅ Total de modelos entrenados: {len(models)}")

    return models


def evaluate_models(models, X_test, y_test, target_columns):
    """Evalúa los modelos en el conjunto de test."""
    print("\n" + "="*70)
    print("EVALUANDO MODELOS")
    print("="*70)

    results = {}

    for target in target_columns:
        print(f"\n{'='*70}")
        print(f"EVALUACIÓN: {target}")
        print("="*70)

        model = models[target]
        y_pred = model.predict(X_test)

        # Métricas
        accuracy = accuracy_score(y_test[target], y_pred)
        f1 = f1_score(y_test[target], y_pred, zero_division=0)

        print(f"\nMétricas:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")

        # Classification report
        print(f"\nClassification Report:")
        report = classification_report(y_test[target], y_pred, zero_division=0)
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_test[target], y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")

        # Guardar resultados
        results[target] = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }

    # Guardar reporte JSON
    report_path = REPORTS_DIR / 'evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Reporte guardado en: {report_path}")

    # Resumen general
    print("\n" + "="*70)
    print("RESUMEN GENERAL")
    print("="*70)

    avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
    avg_f1 = np.mean([r['f1_score'] for r in results.values()])

    print(f"\nPromedio de métricas:")
    print(f"  Accuracy promedio: {avg_accuracy:.4f}")
    print(f"  F1-Score promedio: {avg_f1:.4f}")

    return results


def save_training_info(feature_columns, target_columns, results):
    """Guarda información del entrenamiento."""
    print("\n" + "="*70)
    print("GUARDANDO INFORMACIÓN DEL ENTRENAMIENTO")
    print("="*70)

    info = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'dataset_path': str(DATA_PATH),
        'test_size': TEST_SIZE,
        'random_state': RANDOM_STATE,
        'features': feature_columns,
        'targets': target_columns,
        'num_models': len(target_columns),
        'model_type': 'XGBoost',
        'average_accuracy': float(np.mean([r['accuracy'] for r in results.values()])),
        'average_f1_score': float(np.mean([r['f1_score'] for r in results.values()]))
    }

    info_path = MODELS_DIR / 'training_info.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n✅ Información guardada en: {info_path}")


def main():
    """Pipeline principal de entrenamiento."""
    print("\n" + "🎵" * 35)
    print("ENTRENAMIENTO DE MODELO - URSINGER ML")
    print("🎵" * 35 + "\n")

    # 1. Cargar datos
    df = load_data()

    # 2. Preparar features y targets
    X, y, feature_columns, target_columns = prepare_features_and_targets(df)

    # 3. Split train/test
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Normalizar
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    # 5. Entrenar modelos
    models = train_models(X_train_scaled, y_train, target_columns)

    # 6. Evaluar
    results = evaluate_models(models, X_test_scaled, y_test, target_columns)

    # 7. Guardar info
    save_training_info(feature_columns, target_columns, results)

    print("\n" + "="*70)
    print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"\nModelos guardados en: {MODELS_DIR}")
    print(f"Scalers guardados en: {SCALERS_DIR}")
    print(f"Reportes guardados en: {REPORTS_DIR}")
    print("\n✅ Todo listo para hacer predicciones!")


if __name__ == "__main__":
    main()

