# Cambios posteriores a revision 4

Fecha: 2026-10-27

## Objetivo de este documento

Este archivo indica exactamente:
1. Que se cambio en codigo para atender la observacion 4.
2. Como reproducir, verificar y reutilizar la evidencia generada.

## 1) Cambios tecnicos implementados

### 1.1 Estudio de ablacion en el entrenamiento

Se agrego un modo de ejecucion dedicado a comparaciones controladas:
- CLI: --mode ablation
- Funcion principal: run_ablation_study(...)

Referencias:
- [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L560)
- [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L1102)

### 1.2 Heuristic rule-threshold baseline

Se implemento un baseline interpretable por reglas/umbrales para comparacion controlada.
Nota: los umbrales son referencia tecnica y no thresholds clinicos/diagnosticos.

Referencias:
- [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L83)
- [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L521)

### 1.3 Comparaciones bajo mismas condiciones

Todos los escenarios usan los mismos folds de GroupKFold por group_id/singer_id.

Evidencia:
- [models/reports/ablation_report.json](../models/reports/ablation_report.json)
- Campo: evaluation_protocol

Transparencia adicional implementada:
- Campo: test_set_transparency_note
- Campo por fold: test_source_composition (original_rows, synthetic_rows, synthetic_ratio)

### 1.4 Ajuste importante para dataset legacy

Se corrigio la identificacion de filas sinteticas cuando source_type no distinguia correctamente originales vs sinteticas.
Esto era necesario para que la comparacion con/sin sinteticos fuera real y no aparente.

Referencia:
- [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L571)

## 2) Escenarios evaluados (nomenclatura oficial)

Usar exactamente estos nombres para evitar ambiguedad:
1. xgboost_with_synthetic_with_gender
2. xgboost_without_synthetic_with_gender
3. xgboost_with_synthetic_without_gender
4. xgboost_without_synthetic_without_gender
5. heuristic_rule_threshold_baseline

## 3) Numeros a colocar en resultados

Fuente unica:
- [models/reports/ablation_report.json](../models/reports/ablation_report.json)

Resumen (promedio CV):
- XGBoost + sintetico + genero: accuracy=0.9581, f1=0.8421, micro-f1=0.8336, macro-f1=0.8421, hamming=0.0419, subset=0.8071
- XGBoost - sintetico + genero: accuracy=0.8818, f1=0.0000, micro-f1=0.0000, macro-f1=0.0000, hamming=0.1182, subset=0.4089
- XGBoost + sintetico - genero: accuracy=0.9581, f1=0.8421, micro-f1=0.8336, macro-f1=0.8421, hamming=0.0419, subset=0.8071
- XGBoost - sintetico - genero: accuracy=0.8818, f1=0.0000, micro-f1=0.0000, macro-f1=0.0000, hamming=0.1182, subset=0.4089
- Heuristic rule-threshold baseline: accuracy=0.9196, f1=0.4262, micro-f1=0.4850, macro-f1=0.4262, hamming=0.0804, subset=0.5982

## 4) Hallazgos tecnicos que deben mantenerse en cualquier reporte

### 4.1 Protocolo de comparacion

Mantener siempre estas condiciones:
- Se hizo una ablacion con cinco escenarios.
- Se usaron particiones GroupKFold identicas en todos los escenarios para comparabilidad estricta.
- Esta ablacion corre sobre benchmark aumentado/sintetico, por lo que test puede incluir sinteticos.

### 4.2 Metricas minimas a comunicar

Incluir siempre estas metricas por escenario:
- Accuracy
- F1
- Micro-F1
- Macro-F1
- Hamming loss
- Subset accuracy

### 4.3 Conclusiones obligatorias de la ablacion

Incluir explicitamente, aunque no favorezca al modelo:
1. El baseline simple funciona de manera intermedia, pero queda por debajo de XGBoost con sinteticos.
2. Sin sinteticos, el rendimiento en deteccion positiva cae drasticamente.
3. Retirar genero no cambia el rendimiento global en este dataset.

### 4.4 Limitaciones

Agregar nota de limitacion:
- El efecto nulo de genero podria depender de composicion muestral y balance de clases.
- Conviene validarlo en cohortes mas grandes y heterogeneas.

## 5) Comando para reproducir la evidencia

1. .venv311/bin/python src/ursinger_ml/models/train.py --mode ablation --data-path data/training_data_augmented.csv --cv-splits 5

## 6) Checklist rapido de verificacion interna

1. Confirmar que cualquier tabla/resumen coincide con [models/reports/ablation_report.json](../models/reports/ablation_report.json).
2. Confirmar que se menciona comparabilidad de folds (mismo GroupKFold).
3. Confirmar que se reporta el resultado no favorable (escenarios sin sinteticos).
4. Confirmar que se incluye baseline simple por umbrales.
5. Confirmar que se reporta composicion de test por fold (original/sintetico).
