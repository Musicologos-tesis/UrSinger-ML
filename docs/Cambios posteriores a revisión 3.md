# Cambios posteriores a revision 3 - Guia tecnica interna

Fecha: 2026-10-27

Objetivo de este documento:
- Explicar que se cambio en codigo para atender la observacion 3.
- Indicar como validar y reutilizar los nuevos reportes de metricas.

## 1) Que se cambio en codigo

Archivo modificado principal:
- [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py)

Cambios tecnicos realizados:
1. Metricas por target ampliadas:
- accuracy, precision, recall, F1
- support_positive, support_negative
- false_negatives, false_positives, TP, TN
- confusion_matrix
Referencia: [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L418)

2. Metricas multietiqueta agregadas:
- micro/macro F1
- micro/macro precision
- micro/macro recall
- Hamming loss
- subset accuracy
Referencia: [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L438)

3. Definicion explicita de exactitud agregada incorporada en reportes:
- aggregated_accuracy_definition
Referencia: [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L716)

4. Resultados por fold con variabilidad:
- std y IC95 (semiancho) para accuracy/F1/PR-AUC
Referencia: [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L646)

5. Analisis de FN por target en CV:
- error_analysis por modelo y target
Referencia: [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L691)

6. Holdout extendido con bloque multietiqueta y exactitud agregada:
Referencia: [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L732)

## 2) Artefactos nuevos/actualizados que sustentan la observacion 3

1. CV ampliado:
- [models/reports/cv_report.json](../models/reports/cv_report.json)

2. Holdout ampliado:
- [models/reports/evaluation_report.json](../models/reports/evaluation_report.json)

3. Compatibilidad del script de figuras con nuevo formato de reporte:
- [generate_presentation_visuals.py](../generate_presentation_visuals.py#L57)
- [generate_presentation_visuals.py](../generate_presentation_visuals.py#L220)
- [generate_presentation_visuals.py](../generate_presentation_visuals.py#L327)

## 3) Numeros clave para reportes internos

Fuente: [models/reports/evaluation_report.json](../models/reports/evaluation_report.json)

Por dimension (xgboost holdout):
1. G1: precision 1.0000, recall 0.5000, F1 0.6667, support+ 10, FN 5
2. G2: precision 1.0000, recall 0.4000, F1 0.5714, support+ 10, FN 6
3. G3: precision 0.9000, recall 0.9000, F1 0.9000, support+ 10, FN 1
4. G4: precision 0.9000, recall 0.9000, F1 0.9000, support+ 10, FN 1
5. G5: precision 1.0000, recall 0.9000, F1 0.9474, support+ 10, FN 1

Multietiqueta holdout:
1. micro-F1: 0.8182
2. macro-F1: 0.7971
3. Hamming loss: 0.0457
4. subset accuracy: 0.7857
5. exactitud agregada: 0.9543

Fuente: [models/reports/cv_report.json](../models/reports/cv_report.json)

CV xgboost (5 folds):
1. Accuracy: 0.9581 +/- 0.0169 (DE), IC95 semiancho 0.0148
2. F1: 0.8421 +/- 0.0380 (DE), IC95 semiancho 0.0333
3. PR-AUC: 0.9421 +/- 0.0150 (DE), IC95 semiancho 0.0131
4. Micro-F1 promedio: 0.8336
5. Macro-F1 promedio: 0.8421
6. Hamming loss promedio: 0.0419
7. Subset accuracy promedio: 0.8071

FN en CV (xgboost):
1. G1: 5/29 (17.24%), recall medio 0.8229
2. G2: 4/29 (13.79%), recall medio 0.8762

## 4) Hallazgos tecnicos que deben mantenerse

1. No reportar solo accuracy.
2. Incluir precision, recall, F1 y soporte por dimension.
3. Incluir variabilidad por fold (DE e IC95) en CV.
4. Incluir metricas multietiqueta.
5. Incluir explicacion del trade-off: precision alta pero recall mas bajo en G1/G2.
6. Mantener visible el analisis de FN para G1/G2.

## 5) Texto tecnico breve reutilizable (interno)

Beyond overall accuracy, we report per-dimension precision, recall, F1-score, and class support. This expanded analysis reveals that G1 and G2 are the most challenging dimensions, with lower recall and higher false-negative counts than G3-G5. We also report fold-level variability (standard deviation and 95% confidence intervals) and multilabel metrics (micro-F1, macro-F1, Hamming loss, and subset accuracy), providing a more robust and transparent performance characterization.

## 6) Checklist operativo para tu companero

1. Verificar que se agreguen precision/recall/F1/support por dimension en Results.
2. Verificar que se reporte variabilidad de CV (DE o IC95).
3. Verificar que se incluyan metricas multietiqueta.
4. Verificar que G1/G2 queden discutidos como focos de FN.
5. Verificar que Summary/Discussion/Conclusions queden consistentes con los nuevos numeros.
