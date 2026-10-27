"""
Entrenamiento del modelo XGBoost para clasificación de debilidades vocales.

Este módulo entrena modelos multi-output para predecir debilidades en 5 grupos de habilidad:
- G1: Soporte respiratorio y control de aire
- G2: Afinación y oído tonal
- G3: Estabilidad y vibrato controlado
- G4: Potencia y control dinámico
- G5: Rango y flexibilidad vocal
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    precision_recall_fscore_support,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
import json

# Configuración
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
TRAIN_AUGMENT_RATIO = 0.3

# Rutas
DATA_PATH = Path(__file__).parent.parent.parent.parent / 'data' / 'training_data_with_skill_groups.csv'
MODELS_DIR = Path(__file__).parent.parent.parent.parent / 'models' / 'saved'
SCALERS_DIR = Path(__file__).parent.parent.parent.parent / 'models' / 'scalers'
REPORTS_DIR = Path(__file__).parent.parent.parent.parent / 'models' / 'reports'

# Crear directorios si no existen
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCALERS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

XGB_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': RANDOM_STATE,
    'use_label_encoder': False
}

TARGET_COLUMNS = [
    'weak_G1',
    'weak_G2',
    'weak_G3',
    'weak_G4',
    'weak_G5'
]

FEATURE_COLUMNS = [
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

PHYSICAL_WEAKNESS_RULES = {
    'weak_G1': lambda d: d['durationSec'] < 1.2,
    'weak_G2': lambda d: d['precisionCents'] > 50,
    'weak_G3': lambda d: d['stabilityCents'] > 50,
    'weak_G4': lambda d: d['dynamicRangeDb'] < 10.0,
    'weak_G5': lambda d: d['rangeSpanSemitones'] < 6,
}


def load_data():
    """Carga el dataset aumentado."""
    print("="*70)
    print("CARGANDO DATOS")
    print("="*70)

    df = pd.read_csv(DATA_PATH)
    print(f"\nDataset cargado: {df.shape}")
    print(f"Columnas: {list(df.columns)}")

    return df


def ensure_group_id(df: pd.DataFrame) -> pd.DataFrame:
    """Garantiza una columna group_id para splits sin fuga."""
    df = df.copy()
    if 'group_id' in df.columns:
        return df
    if 'singer_id' in df.columns:
        df['group_id'] = df['singer_id'].astype(str)
        return df
    raise ValueError("No se encontro columna group_id ni singer_id para agrupar.")


def add_source_type_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Marca origen de cada fila para auditar que test permanezca sin augmentacion."""
    df = df.copy()
    if 'source_type' not in df.columns:
        df['source_type'] = 'original'
    return df


def _sample_by_gender(df: pd.DataFrame, ratio: float, random_state: int) -> pd.DataFrame:
    sampled_parts = []
    rng = np.random.RandomState(random_state)
    for gender_value, part in df.groupby('gender'):
        n = int(len(part) * ratio)
        if n <= 0:
            continue
        sampled_parts.append(part.sample(n=n, random_state=rng.randint(0, 10_000_000)))

    if not sampled_parts:
        return df.iloc[0:0].copy()

    return pd.concat(sampled_parts, ignore_index=False)


def _degrade_for_G1(row: pd.Series) -> pd.Series:
    new_row = row.copy()
    new_row['meanRmsDb'] = row['meanRmsDb'] - np.random.uniform(5, 12)
    new_row['rmsConsistency'] = row['rmsConsistency'] + np.random.uniform(5, 10)
    new_row['dynamicRangeDb'] = max(20, row['dynamicRangeDb'] - np.random.uniform(15, 25))
    new_row['stabilityCents'] = row['stabilityCents'] + np.random.uniform(5, 12)
    new_row['durationSec'] = max(0.5, row['durationSec'] * np.random.uniform(0.4, 0.7))
    new_row['weak_G1'] = 1
    return new_row


def _degrade_for_G2(row: pd.Series) -> pd.Series:
    new_row = row.copy()
    new_row['precisionCents'] = row['precisionCents'] + np.random.uniform(15, 30)
    new_row['stabilityCents'] = row['stabilityCents'] + np.random.uniform(8, 15)
    new_row['rangeSpanSemitones'] = max(8, row['rangeSpanSemitones'] - np.random.uniform(3, 7))
    new_row['rangeMinMidi'] = row['rangeMinMidi'] + np.random.uniform(1, 3)
    new_row['rangeMaxMidi'] = row['rangeMaxMidi'] - np.random.uniform(1, 3)
    new_row['meanRmsDb'] = row['meanRmsDb'] - np.random.uniform(2, 5)
    new_row['weak_G2'] = 1
    return new_row


def _degrade_for_G3(row: pd.Series) -> pd.Series:
    new_row = row.copy()
    new_row['stabilityCents'] = row['stabilityCents'] + np.random.uniform(10, 20)
    new_row['rmsConsistency'] = row['rmsConsistency'] + np.random.uniform(7, 12)
    new_row['attackLatencyMs'] = row['attackLatencyMs'] + np.random.uniform(40, 80)
    new_row['meanRmsDb'] = row['meanRmsDb'] - np.random.uniform(3, 8)
    new_row['weak_G3'] = 1
    return new_row


def _degrade_for_G4(row: pd.Series) -> pd.Series:
    new_row = row.copy()
    new_row['meanRmsDb'] = row['meanRmsDb'] - np.random.uniform(8, 15)
    new_row['rmsConsistency'] = row['rmsConsistency'] + np.random.uniform(6, 11)
    new_row['dynamicRangeDb'] = max(3, row['dynamicRangeDb'] - np.random.uniform(20, 55))
    new_row['stabilityCents'] = row['stabilityCents'] + np.random.uniform(6, 12)
    new_row['weak_G4'] = 1
    return new_row


def _degrade_for_G5(row: pd.Series) -> pd.Series:
    new_row = row.copy()
    new_row['rangeMinMidi'] = row['rangeMinMidi'] + np.random.uniform(4, 8)
    new_row['rangeMaxMidi'] = row['rangeMaxMidi'] - np.random.uniform(4, 8)
    new_row['rangeSpanSemitones'] = max(2, row['rangeSpanSemitones'] - np.random.uniform(12, 18))
    new_row['stabilityCents'] = row['stabilityCents'] + np.random.uniform(7, 14)
    new_row['meanRmsDb'] = row['meanRmsDb'] - np.random.uniform(4, 9)
    new_row['rmsConsistency'] = row['rmsConsistency'] + np.random.uniform(5, 9)
    new_row['weak_G5'] = 1
    return new_row


def augment_training_rows(train_df: pd.DataFrame, ratio: float = TRAIN_AUGMENT_RATIO, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Aplica augmentacion solo al train split y conserva el mismo group_id."""
    np.random.seed(random_state)

    original_mask = train_df[TARGET_COLUMNS].sum(axis=1) == 0
    candidate_rows = train_df.loc[original_mask].copy()
    sampled_rows = _sample_by_gender(candidate_rows, ratio=ratio, random_state=random_state)

    if sampled_rows.empty:
        return train_df

    degradation_functions = [
        _degrade_for_G1,
        _degrade_for_G2,
        _degrade_for_G3,
        _degrade_for_G4,
        _degrade_for_G5
    ]

    synthetic_rows = []
    for _, row in sampled_rows.iterrows():
        for degrade_func in degradation_functions:
            degraded = degrade_func(row)
            degraded['group_id'] = row['group_id']
            degraded['source_type'] = 'synthetic'
            synthetic_rows.append(degraded)

    if not synthetic_rows:
        return train_df

    synthetic_df = pd.DataFrame(synthetic_rows)
    return pd.concat([train_df, synthetic_df], ignore_index=True)


def prepare_features_and_targets(
    df,
    return_groups: bool = False,
    save_encoder: bool = True,
    feature_columns_override=None,
):
    """Separa features (X) y targets (y)."""
    print("\n" + "="*70)
    print("PREPARANDO FEATURES Y TARGETS")
    print("="*70)

    feature_columns = feature_columns_override if feature_columns_override is not None else FEATURE_COLUMNS
    target_columns = TARGET_COLUMNS

    print(f"\nFeatures seleccionados ({len(feature_columns)}):")
    for feat in feature_columns:
        print(f"  - {feat}")

    print(f"\nTargets ({len(target_columns)}):")
    for target in target_columns:
        count = df[target].sum()
        print(f"  - {target}: {count} positivos ({count/len(df)*100:.1f}%)")

    X = df[feature_columns].copy()
    y = df[target_columns].copy()

    if 'gender' in X.columns:
        # Codificar genero con mapeo estable para evitar fit con todo el dataset.
        gender_map = {'F': 0, 'M': 1}
        X['gender'] = X['gender'].map(gender_map)
        if X['gender'].isna().any():
            unknown = sorted(df.loc[X['gender'].isna(), 'gender'].astype(str).unique())
            raise ValueError(f"Valores de genero no soportados: {unknown}")
        X['gender'] = X['gender'].astype(int)

    # Guardar el encoder
    if save_encoder and 'gender' in feature_columns:
        le = LabelEncoder()
        le.fit(['F', 'M'])
        joblib.dump(le, SCALERS_DIR / 'gender_encoder.pkl')
        print(f"\n✅ Gender encoder guardado en: {SCALERS_DIR / 'gender_encoder.pkl'}")

    if return_groups:
        groups = df['group_id'] if 'group_id' in df.columns else (df['singer_id'] if 'singer_id' in df.columns else None)
        return X, y, feature_columns, target_columns, groups

    return X, y, feature_columns, target_columns


def split_data(X, y, groups=None, strategy: str = "random"):
    """Divide datos en train/test."""
    print("\n" + "="*70)
    print("DIVIDIENDO DATOS (TRAIN/TEST)")
    print("="*70)

    if strategy == "group":
        if groups is None:
            raise ValueError("Se requiere groups para split por cantante.")

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print("\nSplit por cantante activado (GroupShuffleSplit)")
        print(f"Cantantes en train: {groups.iloc[train_idx].nunique()}")
        print(f"Cantantes en test: {groups.iloc[test_idx].nunique()}")
    else:
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
    print(f"\nParámetros de XGBoost:")
    for key, value in XGB_PARAMS.items():
        print(f"  {key}: {value}")

    for target in target_columns:
        print(f"\n📊 Entrenando modelo para {target}...")

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train[target])

        models[target] = model

        # Guardar modelo
        model_path = MODELS_DIR / f'xgb_{target}.pkl'
        joblib.dump(model, model_path)
        print(f"   ✅ Modelo guardado en: {model_path}")

    print(f"\n✅ Total de modelos entrenados: {len(models)}")

    return models


def _predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.ravel()

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.ravel(scores)

    return None


def _safe_average_precision(y_true, y_scores):
    if y_scores is None:
        return None
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_scores))


def _mean_ignore_none(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(np.mean(valid))


def _std_ignore_none(values):
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return None
    return float(np.std(valid, ddof=1))


def _ci95_halfwidth(values):
    valid = [v for v in values if v is not None]
    if len(valid) < 2:
        return None
    std = np.std(valid, ddof=1)
    return float(1.96 * (std / np.sqrt(len(valid))))


def _safe_metric(func, *args, **kwargs):
    try:
        return float(func(*args, **kwargs))
    except Exception:
        return None


def _compute_binary_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    support_pos = int(tp + fn)
    support_neg = int(tn + fp)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'support_positive': support_pos,
        'support_negative': support_neg,
        'false_negatives': int(fn),
        'false_positives': int(fp),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'confusion_matrix': cm.tolist()
    }


def _compute_multilabel_metrics(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame):
    y_true = y_true_df.values
    y_pred = y_pred_df.values
    return {
        'micro_f1': _safe_metric(f1_score, y_true, y_pred, average='micro', zero_division=0),
        'macro_f1': _safe_metric(f1_score, y_true, y_pred, average='macro', zero_division=0),
        'micro_precision': _safe_metric(precision_score, y_true, y_pred, average='micro', zero_division=0),
        'macro_precision': _safe_metric(precision_score, y_true, y_pred, average='macro', zero_division=0),
        'micro_recall': _safe_metric(recall_score, y_true, y_pred, average='micro', zero_division=0),
        'macro_recall': _safe_metric(recall_score, y_true, y_pred, average='macro', zero_division=0),
        'hamming_loss': _safe_metric(hamming_loss, y_true, y_pred),
        'subset_accuracy': _safe_metric(accuracy_score, y_true, y_pred),
        'label_wise_accuracy_mean': _safe_metric(lambda a, b: np.mean(np.mean(a == b, axis=0)), y_true, y_pred)
    }


def _compute_aggregated_label_accuracy(per_target_metrics):
    total_correct = 0
    total_labels = 0
    for metrics in per_target_metrics.values():
        tn = metrics['true_negatives']
        tp = metrics['true_positives']
        fp = metrics['false_positives']
        fn = metrics['false_negatives']
        total_correct += tn + tp
        total_labels += tn + tp + fp + fn
    if total_labels == 0:
        return None
    return float(total_correct / total_labels)


def _compute_dataset_metrics(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame, target_columns):
    per_target = {}
    for target in target_columns:
        per_target[target] = _compute_binary_metrics(y_true_df[target], y_pred_df[target])
    multilabel = _compute_multilabel_metrics(y_true_df[target_columns], y_pred_df[target_columns])
    aggregated_label_accuracy = _compute_aggregated_label_accuracy(per_target)
    avg_accuracy = float(np.mean([m['accuracy'] for m in per_target.values()]))
    avg_f1 = float(np.mean([m['f1_score'] for m in per_target.values()]))
    return {
        'per_target': per_target,
        'multilabel': multilabel,
        'aggregated_label_accuracy': aggregated_label_accuracy,
        'average': {
            'accuracy': avg_accuracy,
            'f1_score': avg_f1,
            'pr_auc': None,
        },
    }


def _fit_predict_xgboost_per_target(X_train, y_train, X_test, target_columns):
    pred_matrix = {}
    for target in target_columns:
        y_tr = y_train[target]
        if y_tr.nunique() < 2:
            # Cuando no hay dos clases en train, usar clasificador constante.
            constant_value = int(y_tr.iloc[0])
            pred_matrix[target] = np.full(len(X_test), constant_value)
            continue
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_tr)
        pred_matrix[target] = model.predict(X_test)
    return pd.DataFrame(pred_matrix, index=X_test.index)


def _predict_with_physical_rules(df: pd.DataFrame, target_columns):
    preds = {}
    for target in target_columns:
        if target not in PHYSICAL_WEAKNESS_RULES:
            raise ValueError(f"No hay regla fisica para {target}")
        preds[target] = PHYSICAL_WEAKNESS_RULES[target](df).astype(int)
    return pd.DataFrame(preds, index=df.index)


def _summarize_ablation_scenarios(scenarios):
    summary = {}
    for name, folds in scenarios.items():
        avg_acc = _mean_ignore_none([f['metrics']['average']['accuracy'] for f in folds])
        std_acc = _std_ignore_none([f['metrics']['average']['accuracy'] for f in folds])
        ci_acc = _ci95_halfwidth([f['metrics']['average']['accuracy'] for f in folds])
        avg_f1 = _mean_ignore_none([f['metrics']['average']['f1_score'] for f in folds])
        std_f1 = _std_ignore_none([f['metrics']['average']['f1_score'] for f in folds])
        ci_f1 = _ci95_halfwidth([f['metrics']['average']['f1_score'] for f in folds])
        micro_f1 = _mean_ignore_none([f['metrics']['multilabel'].get('micro_f1') for f in folds])
        macro_f1 = _mean_ignore_none([f['metrics']['multilabel'].get('macro_f1') for f in folds])
        hamming = _mean_ignore_none([f['metrics']['multilabel'].get('hamming_loss') for f in folds])
        subset = _mean_ignore_none([f['metrics']['multilabel'].get('subset_accuracy') for f in folds])

        summary[name] = {
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'ci95_halfwidth_accuracy': ci_acc,
            'avg_f1_score': avg_f1,
            'std_f1_score': std_f1,
            'ci95_halfwidth_f1_score': ci_f1,
            'aggregated_label_accuracy': _mean_ignore_none([f['metrics']['aggregated_label_accuracy'] for f in folds]),
            'multilabel_avg': {
                'micro_f1': micro_f1,
                'macro_f1': macro_f1,
                'hamming_loss': hamming,
                'subset_accuracy': subset,
            },
            'num_folds': len(folds),
        }
    return summary


def run_ablation_study(
    df: pd.DataFrame,
    n_splits: int = CV_SPLITS,
    include_rule_baseline: bool = True,
):
    """Ejecuta comparaciones/ablaciones bajo condiciones de evaluacion equivalentes."""
    df = ensure_group_id(add_source_type_if_missing(df))
    groups = df['group_id']

    # Si source_type fue agregado por defecto (todo original), inferir sinteticos
    # para el dataset legacy augmentado donde positivos fueron generados sinteticamente.
    if 'source_type' in df.columns and (df['source_type'] == 'original').all():
        inferred_synthetic = df[TARGET_COLUMNS].sum(axis=1) > 0
        if inferred_synthetic.any():
            df.loc[inferred_synthetic, 'source_type'] = 'synthetic'

    target_columns = TARGET_COLUMNS
    feature_with_gender = FEATURE_COLUMNS
    feature_no_gender = [c for c in FEATURE_COLUMNS if c != 'gender']

    gkf = GroupKFold(n_splits=n_splits)
    scenarios = {
        'xgboost_with_synthetic_with_gender': [],
        'xgboost_without_synthetic_with_gender': [],
        'xgboost_with_synthetic_without_gender': [],
        'xgboost_without_synthetic_without_gender': [],
    }
    if include_rule_baseline:
        scenarios['rule_threshold_baseline'] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, df['weak_G1'], groups), start=1):
        train_df_full = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)

        train_df_no_syn = train_df_full[train_df_full['source_type'] == 'original'].copy().reset_index(drop=True)

        # Escenario 1: con sinteticos + con genero
        X_train, y_train, _, _ = prepare_features_and_targets(
            train_df_full,
            return_groups=False,
            save_encoder=False,
            feature_columns_override=feature_with_gender,
        )
        X_test, y_test, _, _ = prepare_features_and_targets(
            test_df,
            return_groups=False,
            save_encoder=False,
            feature_columns_override=feature_with_gender,
        )
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        y_pred_df = _fit_predict_xgboost_per_target(X_train_scaled, y_train, X_test_scaled, target_columns)
        metrics = _compute_dataset_metrics(y_test, y_pred_df, target_columns)
        scenarios['xgboost_with_synthetic_with_gender'].append({'fold': fold_idx, 'num_train': int(len(train_df_full)), 'num_test': int(len(test_df)), 'metrics': metrics})

        # Escenario 2: sin sinteticos + con genero
        X_train_ns, y_train_ns, _, _ = prepare_features_and_targets(
            train_df_no_syn,
            return_groups=False,
            save_encoder=False,
            feature_columns_override=feature_with_gender,
        )
        scaler_ns = StandardScaler()
        X_train_ns_scaled = pd.DataFrame(scaler_ns.fit_transform(X_train_ns), columns=X_train_ns.columns, index=X_train_ns.index)
        X_test_ns_scaled = pd.DataFrame(scaler_ns.transform(X_test), columns=X_test.columns, index=X_test.index)
        y_pred_ns_df = _fit_predict_xgboost_per_target(X_train_ns_scaled, y_train_ns, X_test_ns_scaled, target_columns)
        metrics_ns = _compute_dataset_metrics(y_test, y_pred_ns_df, target_columns)
        scenarios['xgboost_without_synthetic_with_gender'].append({'fold': fold_idx, 'num_train': int(len(train_df_no_syn)), 'num_test': int(len(test_df)), 'metrics': metrics_ns})

        # Escenario 3: con sinteticos + sin genero
        X_train_ng, y_train_ng, _, _ = prepare_features_and_targets(
            train_df_full,
            return_groups=False,
            save_encoder=False,
            feature_columns_override=feature_no_gender,
        )
        X_test_ng, y_test_ng, _, _ = prepare_features_and_targets(
            test_df,
            return_groups=False,
            save_encoder=False,
            feature_columns_override=feature_no_gender,
        )
        scaler_ng = StandardScaler()
        X_train_ng_scaled = pd.DataFrame(scaler_ng.fit_transform(X_train_ng), columns=X_train_ng.columns, index=X_train_ng.index)
        X_test_ng_scaled = pd.DataFrame(scaler_ng.transform(X_test_ng), columns=X_test_ng.columns, index=X_test_ng.index)
        y_pred_ng_df = _fit_predict_xgboost_per_target(X_train_ng_scaled, y_train_ng, X_test_ng_scaled, target_columns)
        metrics_ng = _compute_dataset_metrics(y_test_ng, y_pred_ng_df, target_columns)
        scenarios['xgboost_with_synthetic_without_gender'].append({'fold': fold_idx, 'num_train': int(len(train_df_full)), 'num_test': int(len(test_df)), 'metrics': metrics_ng})

        # Escenario 4: sin sinteticos + sin genero
        X_train_ns_ng, y_train_ns_ng, _, _ = prepare_features_and_targets(
            train_df_no_syn,
            return_groups=False,
            save_encoder=False,
            feature_columns_override=feature_no_gender,
        )
        scaler_ns_ng = StandardScaler()
        X_train_ns_ng_scaled = pd.DataFrame(scaler_ns_ng.fit_transform(X_train_ns_ng), columns=X_train_ns_ng.columns, index=X_train_ns_ng.index)
        X_test_ns_ng_scaled = pd.DataFrame(scaler_ns_ng.transform(X_test_ng), columns=X_test_ng.columns, index=X_test_ng.index)
        y_pred_ns_ng_df = _fit_predict_xgboost_per_target(X_train_ns_ng_scaled, y_train_ns_ng, X_test_ns_ng_scaled, target_columns)
        metrics_ns_ng = _compute_dataset_metrics(y_test_ng, y_pred_ns_ng_df, target_columns)
        scenarios['xgboost_without_synthetic_without_gender'].append({'fold': fold_idx, 'num_train': int(len(train_df_no_syn)), 'num_test': int(len(test_df)), 'metrics': metrics_ns_ng})

        # Baseline por reglas
        if include_rule_baseline:
            y_rule_pred = _predict_with_physical_rules(test_df, target_columns)
            rule_metrics = _compute_dataset_metrics(y_test, y_rule_pred, target_columns)
            scenarios['rule_threshold_baseline'].append({'fold': fold_idx, 'num_train': int(len(train_df_full)), 'num_test': int(len(test_df)), 'metrics': rule_metrics})

    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'dataset_path': str(DATA_PATH),
        'evaluation_protocol': 'GroupKFold by group_id/singer_id with identical folds across all scenarios',
        'targets': target_columns,
        'scenarios': scenarios,
        'summary': _summarize_ablation_scenarios(scenarios),
    }

    report_path = REPORTS_DIR / 'ablation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Reporte de ablacion guardado en: {report_path}")
    return report


def get_model_factories():
    return {
        'log_reg': lambda: LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        'random_forest': lambda: RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        ),
        'xgboost': lambda: xgb.XGBClassifier(**XGB_PARAMS)
    }


def cross_validate_models(X, y, groups, target_columns, n_splits: int = CV_SPLITS):
    """Cross-validation por cantante con modelos baseline y XGBoost."""
    if groups is None:
        raise ValueError("Se requiere groups para GroupKFold.")

    gkf = GroupKFold(n_splits=n_splits)
    model_factories = get_model_factories()

    results = {model_name: [] for model_name in model_factories}

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y['weak_G1'], groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        for model_name, factory in model_factories.items():
            per_target = {}
            pred_matrix = {}

            for target in target_columns:
                model = factory()
                model.fit(X_train_scaled, y_train[target])

                y_pred = model.predict(X_test_scaled)
                y_scores = _predict_scores(model, X_test_scaled)

                metrics = _compute_binary_metrics(y_test[target], y_pred)
                metrics['pr_auc'] = _safe_average_precision(y_test[target], y_scores)
                per_target[target] = metrics
                pred_matrix[target] = y_pred

            y_pred_df = pd.DataFrame(pred_matrix, index=y_test.index)
            multilabel = _compute_multilabel_metrics(y_test[target_columns], y_pred_df[target_columns])
            aggregated_label_accuracy = _compute_aggregated_label_accuracy(per_target)

            avg_accuracy = np.mean([m['accuracy'] for m in per_target.values()])
            avg_f1 = np.mean([m['f1_score'] for m in per_target.values()])
            avg_pr = _mean_ignore_none([m['pr_auc'] for m in per_target.values()])

            results[model_name].append({
                'fold': fold_idx + 1,
                'num_train': int(len(train_idx)),
                'num_test': int(len(test_idx)),
                'per_target': per_target,
                'aggregated_label_accuracy': aggregated_label_accuracy,
                'multilabel': multilabel,
                'average': {
                    'accuracy': float(avg_accuracy),
                    'f1_score': float(avg_f1),
                    'pr_auc': avg_pr
                }
            })

    return results


def cross_validate_models_with_train_augmentation(df: pd.DataFrame, target_columns, n_splits: int = CV_SPLITS):
    """Cross-validation por grupos con augmentacion exclusiva en train por fold."""
    df = ensure_group_id(add_source_type_if_missing(df))

    groups = df['group_id']
    gkf = GroupKFold(n_splits=n_splits)
    model_factories = get_model_factories()

    results = {model_name: [] for model_name in model_factories}
    audit_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, df['weak_G1'], groups), start=1):
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)

        train_df = augment_training_rows(train_df, ratio=TRAIN_AUGMENT_RATIO, random_state=RANDOM_STATE + fold_idx)

        X_train, y_train, _, _ = prepare_features_and_targets(train_df, return_groups=False, save_encoder=False)
        X_test, y_test, _, _ = prepare_features_and_targets(test_df, return_groups=False, save_encoder=False)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

        train_groups = set(train_df['group_id'].astype(str).unique())
        test_groups = set(test_df['group_id'].astype(str).unique())
        intersection = sorted(train_groups & test_groups)
        test_aug_rows = int((test_df['source_type'] == 'synthetic').sum())

        audit_rows.append({
            'fold': fold_idx,
            'train_group_count': int(len(train_groups)),
            'test_group_count': int(len(test_groups)),
            'group_intersection_count': int(len(intersection)),
            'group_intersection': intersection,
            'test_augmented_rows': test_aug_rows,
            'status': 'OK' if len(intersection) == 0 and test_aug_rows == 0 else 'Leak'
        })

        for model_name, factory in model_factories.items():
            per_target = {}
            pred_matrix = {}

            for target in target_columns:
                model = factory()
                model.fit(X_train_scaled, y_train[target])

                y_pred = model.predict(X_test_scaled)
                y_scores = _predict_scores(model, X_test_scaled)

                metrics = _compute_binary_metrics(y_test[target], y_pred)
                metrics['pr_auc'] = _safe_average_precision(y_test[target], y_scores)
                per_target[target] = metrics
                pred_matrix[target] = y_pred

            y_pred_df = pd.DataFrame(pred_matrix, index=y_test.index)
            multilabel = _compute_multilabel_metrics(y_test[target_columns], y_pred_df[target_columns])
            aggregated_label_accuracy = _compute_aggregated_label_accuracy(per_target)

            avg_accuracy = np.mean([m['accuracy'] for m in per_target.values()])
            avg_f1 = np.mean([m['f1_score'] for m in per_target.values()])
            avg_pr = _mean_ignore_none([m['pr_auc'] for m in per_target.values()])

            results[model_name].append({
                'fold': fold_idx,
                'num_train': int(len(train_df)),
                'num_test': int(len(test_df)),
                'per_target': per_target,
                'aggregated_label_accuracy': aggregated_label_accuracy,
                'multilabel': multilabel,
                'average': {
                    'accuracy': float(avg_accuracy),
                    'f1_score': float(avg_f1),
                    'pr_auc': avg_pr
                }
            })

    return results, audit_rows


def save_leakage_audit_report(audit_rows, n_splits: int):
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'dataset_path': str(DATA_PATH),
        'cv_splits': n_splits,
        'group_definition': 'group_id (fallback singer_id)',
        'audit': audit_rows
    }

    report_path = REPORTS_DIR / 'leakage_audit_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Auditoria de fuga guardada en: {report_path}")


def summarize_cv_results(results):
    summary = {}
    for model_name, folds in results.items():
        avg_accuracy = _mean_ignore_none([f['average']['accuracy'] for f in folds])
        std_accuracy = _std_ignore_none([f['average']['accuracy'] for f in folds])
        ci95_accuracy = _ci95_halfwidth([f['average']['accuracy'] for f in folds])
        avg_f1 = _mean_ignore_none([f['average']['f1_score'] for f in folds])
        std_f1 = _std_ignore_none([f['average']['f1_score'] for f in folds])
        ci95_f1 = _ci95_halfwidth([f['average']['f1_score'] for f in folds])
        avg_pr = _mean_ignore_none([f['average']['pr_auc'] for f in folds])
        std_pr = _std_ignore_none([f['average']['pr_auc'] for f in folds])
        ci95_pr = _ci95_halfwidth([f['average']['pr_auc'] for f in folds])
        aggregated_label_accuracy = _mean_ignore_none([f.get('aggregated_label_accuracy') for f in folds])
        aggregated_label_accuracy_std = _std_ignore_none([f.get('aggregated_label_accuracy') for f in folds])

        micro_f1 = _mean_ignore_none([f['multilabel'].get('micro_f1') for f in folds])
        macro_f1 = _mean_ignore_none([f['multilabel'].get('macro_f1') for f in folds])
        hamming = _mean_ignore_none([f['multilabel'].get('hamming_loss') for f in folds])
        subset = _mean_ignore_none([f['multilabel'].get('subset_accuracy') for f in folds])

        summary[model_name] = {
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'ci95_halfwidth_accuracy': ci95_accuracy,
            'avg_f1_score': avg_f1,
            'std_f1_score': std_f1,
            'ci95_halfwidth_f1_score': ci95_f1,
            'avg_pr_auc': avg_pr,
            'std_pr_auc': std_pr,
            'ci95_halfwidth_pr_auc': ci95_pr,
            'aggregated_label_accuracy': aggregated_label_accuracy,
            'std_aggregated_label_accuracy': aggregated_label_accuracy_std,
            'multilabel_avg': {
                'micro_f1': micro_f1,
                'macro_f1': macro_f1,
                'hamming_loss': hamming,
                'subset_accuracy': subset,
            },
            'num_folds': len(folds)
        }

    return summary


def save_cv_report(results, feature_columns, target_columns, n_splits: int):
    def _build_error_analysis(model_folds):
        analysis = {}
        for target in target_columns:
            fns = [f['per_target'][target]['false_negatives'] for f in model_folds]
            supports = [f['per_target'][target]['support_positive'] for f in model_folds]
            recalls = [f['per_target'][target]['recall'] for f in model_folds]
            total_fn = int(np.sum(fns))
            total_pos = int(np.sum(supports))
            fn_rate = float(total_fn / total_pos) if total_pos > 0 else None
            analysis[target] = {
                'total_false_negatives': total_fn,
                'total_positive_support': total_pos,
                'false_negative_rate': fn_rate,
                'recall_mean': _mean_ignore_none(recalls),
                'recall_std': _std_ignore_none(recalls)
            }
        return analysis

    error_analysis = {model: _build_error_analysis(folds) for model, folds in results.items()}

    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'dataset_path': str(DATA_PATH),
        'cv_splits': n_splits,
        'random_state': RANDOM_STATE,
        'aggregated_accuracy_definition': 'Total correct binary label decisions across all targets and samples divided by total binary decisions (samples x labels).',
        'features': feature_columns,
        'targets': target_columns,
        'models': list(results.keys()),
        'summary': summarize_cv_results(results),
        'folds': results,
        'error_analysis': error_analysis
    }

    report_path = REPORTS_DIR / 'cv_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Reporte CV guardado en: {report_path}")


def evaluate_models(models, X_test, y_test, target_columns):
    """Evalúa los modelos en el conjunto de test."""
    print("\n" + "="*70)
    print("EVALUANDO MODELOS")
    print("="*70)

    results = {}
    predictions = {}

    for target in target_columns:
        print(f"\n{'='*70}")
        print(f"EVALUACIÓN: {target}")
        print("="*70)

        model = models[target]
        y_pred = model.predict(X_test)
        predictions[target] = y_pred

        # Métricas
        binary_metrics = _compute_binary_metrics(y_test[target], y_pred)
        accuracy = binary_metrics['accuracy']
        f1 = binary_metrics['f1_score']
        precision = binary_metrics['precision']
        recall = binary_metrics['recall']

        print(f"\nMétricas:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Support positiva: {binary_metrics['support_positive']}")

        # Classification report
        print(f"\nClassification Report:")
        report = classification_report(y_test[target], y_pred, zero_division=0)
        print(report)

        # Confusion matrix
        cm = np.array(binary_metrics['confusion_matrix'])
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")

        # Guardar resultados
        results[target] = {
            **binary_metrics,
            'classification_report': report
        }

    y_pred_df = pd.DataFrame(predictions, index=y_test.index)
    multilabel = _compute_multilabel_metrics(y_test[target_columns], y_pred_df[target_columns])
    aggregated_label_accuracy = _compute_aggregated_label_accuracy(results)

    # Guardar reporte JSON
    report_path = REPORTS_DIR / 'evaluation_report.json'
    full_report = {
        'per_target': results,
        'aggregated_accuracy_definition': 'Total correct binary label decisions across all targets and samples divided by total binary decisions (samples x labels).',
        'aggregated_label_accuracy': aggregated_label_accuracy,
        'multilabel_metrics': multilabel,
        'test_size_samples': int(len(y_test))
    }
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2)
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
    print(f"  Aggregated label accuracy: {aggregated_label_accuracy:.4f}")
    print(f"  Micro-F1: {multilabel['micro_f1']:.4f}")
    print(f"  Macro-F1: {multilabel['macro_f1']:.4f}")
    print(f"  Hamming loss: {multilabel['hamming_loss']:.4f}")
    print(f"  Subset accuracy: {multilabel['subset_accuracy']:.4f}")

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


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento y evaluacion de modelos XGBoost")
    parser.add_argument("--mode", choices=["train", "cv", "ablation"], default="train")
    parser.add_argument("--split-strategy", choices=["random", "group"], default="group")
    parser.add_argument("--cv-splits", type=int, default=CV_SPLITS)
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH))
    parser.add_argument("--disable-train-only-augmentation", action="store_true")
    return parser.parse_args()


def main():
    """Pipeline principal de entrenamiento."""
    print("\n" + "🎵" * 35)
    print("ENTRENAMIENTO DE MODELO - URSINGER ML")
    print("🎵" * 35 + "\n")

    args = parse_args()

    global DATA_PATH
    DATA_PATH = Path(args.data_path)

    # 1. Cargar datos
    df = ensure_group_id(add_source_type_if_missing(load_data()))

    # 2. Preparar features y targets
    if args.mode == "ablation":
        run_ablation_study(df, n_splits=args.cv_splits, include_rule_baseline=True)
        print("\nESTUDIO DE ABLACION COMPLETADO")
        return

    if args.mode == "cv":
        if args.disable_train_only_augmentation:
            X, y, feature_columns, target_columns, groups = prepare_features_and_targets(
                df,
                return_groups=True,
                save_encoder=False
            )
            results = cross_validate_models(X, y, groups, target_columns, n_splits=args.cv_splits)
            save_cv_report(results, feature_columns, target_columns, args.cv_splits)
            print("\n⚠️ CV ejecutada sin augmentacion intra-fold (modo legacy)")
            return

        feature_columns = FEATURE_COLUMNS
        target_columns = TARGET_COLUMNS
        results, audit_rows = cross_validate_models_with_train_augmentation(
            df,
            target_columns,
            n_splits=args.cv_splits
        )
        save_cv_report(results, feature_columns, target_columns, args.cv_splits)
        save_leakage_audit_report(audit_rows, args.cv_splits)
        print("\nEVALUACION CV COMPLETADA")
        return

    X, y, feature_columns, target_columns, groups = prepare_features_and_targets(df, return_groups=True)

    if args.split_strategy == "group" and not args.disable_train_only_augmentation:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(df, y['weak_G1'], groups))

        train_df = augment_training_rows(df.iloc[train_idx].copy(), ratio=TRAIN_AUGMENT_RATIO, random_state=RANDOM_STATE)
        test_df = df.iloc[test_idx].copy()

        X_train, y_train, _, _ = prepare_features_and_targets(train_df, return_groups=False)
        X_test, y_test, _, _ = prepare_features_and_targets(test_df, return_groups=False)

        train_groups = set(train_df['group_id'].astype(str).unique())
        test_groups = set(test_df['group_id'].astype(str).unique())
        leakage = train_groups & test_groups
        print(f"\nVerificacion de grupos train/test: interseccion={len(leakage)}")
        print(f"Filas sinteticas en test: {(test_df['source_type'] == 'synthetic').sum()}")
    else:
        # 3. Split train/test
        X_train, X_test, y_train, y_test = split_data(
            X,
            y,
            groups=groups,
            strategy=args.split_strategy
        )

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

