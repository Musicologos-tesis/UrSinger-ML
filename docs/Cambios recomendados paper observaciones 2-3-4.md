# Cambios recomendados al paper (Observaciones 2, 3 y 4)

Fecha: 2026-10-27

## 1) Decision sobre figuras

### 1.1 ¿Es necesario cambiar figuras?
No es estrictamente necesario regenerar figuras para responder las observaciones 2, 3 y 4.

### 1.2 ¿Que figuras pueden quedarse como estan?
1. Figuras 1-6: pueden mantenerse (arquitectura, flujo, composicion del dataset).
2. Figuras 7-9: pueden mantenerse siempre que se explicite que corresponden al benchmark historico aumentado y no a la evaluacion leakage-free final.
3. Figuras 10-13: pueden mantenerse (evaluacion de usuarios, expertos y analisis longitudinal preliminar).

### 1.3 ¿Cuando si convendria cambiar/agregar figura?
Solo si el journal exige evidencia visual de la observacion 4. En ese caso, agregar una figura nueva de barras para ablacion (no reemplazar 7-9). Si no lo exige, una tabla nueva es suficiente.

## 2) Cambios globales obligatorios de redaccion

1. Mantener consistencia terminologica:
- historical augmented benchmark
- leakage-free protocol
- ablation study under identical GroupKFold partitions

2. No presentar resultados historicos como evidencia final leakage-free.

3. Reportar resultados no favorables al modelo cuando correspondan (escenarios sin sinteticos).

## 3) Cambios recomendados por seccion

## 3.1 Abstract

### Problema actual
El abstract reporta metricas globales altas (94.1% accuracy, 77.9% F1, 88.7% PR-AUC) sin separar claramente benchmark historico vs evaluacion leakage-free/ablacion.

### Cambio recomendado
Agregar una frase breve de transparencia metodologica al final del bloque de resultados analiticos.

### Texto sugerido (ingles, listo para pegar)
These cross-validation values correspond to the historical augmented benchmark used for model-comparison during development. In an additional controlled ablation on the same augmented benchmark (with identical singer-group folds across scenarios), performance remained strongly dependent on synthetic augmentation, while removing gender did not materially change aggregate metrics in this dataset.

## 3.2 Section 3.5 Experimental Protocol and Baselines

### Problema actual
La seccion describe GroupKFold y baselines clasicos, pero no deja explicito el nuevo baseline por reglas/umbrales ni la comparacion con/sin sinteticos y con/sin genero bajo particiones identicas.

### Cambios recomendados
1. Agregar un parrafo que describa el estudio de ablacion con cinco escenarios comparables.
2. Incluir el heuristic rule-threshold baseline como referencia interpretable.

### Texto sugerido (ingles, listo para pegar)
In response to reviewer feedback, we added a controlled ablation protocol evaluated under identical GroupKFold partitions by singer/group across all scenarios on the augmented/synthetic development benchmark. We compared: (i) XGBoost with synthetic augmentation and gender, (ii) XGBoost without synthetic augmentation and with gender, (iii) XGBoost with synthetic augmentation and without gender, (iv) XGBoost without synthetic augmentation and without gender, and (v) a heuristic rule-threshold baseline. The heuristic rule-threshold baseline uses fixed acoustic cutoffs aligned with the five pedagogical dimensions and is included as an interpretable non-ML reference, not as a clinical/diagnostic thresholding system.

## 3.3 Section 4.1 Classification Performance (actual)

### Problema actual
La seccion puede interpretarse como resultado final unico, sin marcar suficiente separacion con analisis leakage-free y ablacion.

### Cambio recomendado
Mantener la seccion pero aclarar en el primer parrafo que corresponde al benchmark historico.

### Frase clave a insertar
Results in this subsection correspond to the historical augmented benchmark used for model comparison and should be interpreted together with the leakage-free sensitivity analysis (Obs. 2) and the controlled ablation on the augmented benchmark (Obs. 4) reported below.

## 3.4 Nueva subseccion en Results para observacion 2

Agregar una subseccion nueva despues de 4.1 (o despues de 4.3 si prefieres continuidad con figuras):

### Titulo sugerido
4.X Leakage-Free Sensitivity Analysis

### Contenido minimo
1. Aclarar que en el protocolo corregido la particion por grupos se aplica antes de cualquier augmentacion.
2. Aclarar que la augmentacion ocurre solo en entrenamiento.
3. Reportar evidencia por fold: interseccion de grupos train-test = 0 y sinteticos en test = 0.
4. Explicar por que puede haber accuracy alta con F1 cercano a 0 (composicion de positivos en test).

### Tabla recomendada (agregar)

Table X. Leakage audit summary under corrected protocol (5-fold GroupKFold).

| Fold | Group overlap (train-test) | Synthetic rows in test | Status |
|---|---:|---:|---|
| 1 | 0 | 0 | OK |
| 2 | 0 | 0 | OK |
| 3 | 0 | 0 | OK |
| 4 | 0 | 0 | OK |
| 5 | 0 | 0 | OK |

### Texto sugerido (ingles, breve)
Under the corrected leakage-free protocol, group-based splitting was performed before augmentation, synthetic data were generated only within each training fold, and no synthetic rows appeared in any test fold. Group overlap between train and test remained zero in all folds.

## 3.5 Nueva subseccion en Results para observacion 4

### Titulo sugerido
4.Y Ablation and Baseline Comparison Under Comparable Conditions

### Tabla obligatoria (agregar)

Table Y. Controlled ablation results on the augmented benchmark under identical GroupKFold partitions across scenarios (5 folds).

| Scenario | Accuracy | F1 | Micro-F1 | Macro-F1 | Hamming loss | Subset accuracy |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost + synthetic + gender | 0.9581 | 0.8421 | 0.8336 | 0.8421 | 0.0419 | 0.8071 |
| XGBoost - synthetic + gender | 0.8818 | 0.0000 | 0.0000 | 0.0000 | 0.1182 | 0.4089 |
| XGBoost + synthetic - gender | 0.9581 | 0.8421 | 0.8336 | 0.8421 | 0.0419 | 0.8071 |
| XGBoost - synthetic - gender | 0.8818 | 0.0000 | 0.0000 | 0.0000 | 0.1182 | 0.4089 |
| Heuristic rule-threshold baseline | 0.9196 | 0.4262 | 0.4850 | 0.4262 | 0.0804 | 0.5982 |

Optional transparency table (recommended):

Table Y.1. Test composition by fold in the controlled ablation (augmented benchmark).

| Fold | Original rows in test | Synthetic rows in test | Synthetic ratio |
|---|---:|---:|---:|
| 1 | 15 | 35 | 0.70 |
| 2 | 20 | 30 | 0.60 |
| 3 | 20 | 30 | 0.60 |
| 4 | 25 | 25 | 0.50 |
| 5 | 20 | 25 | 0.56 |

### Texto sugerido (ingles, breve)
The controlled ablation confirms three points under strictly comparable evaluation conditions (identical folds): (1) XGBoost with synthetic augmentation outperforms the heuristic rule-threshold baseline across multilabel metrics; (2) removing synthetic augmentation produces a substantial drop in positive-class detection (near-zero F1); and (3) removing gender does not materially change aggregate performance in this dataset.

## 3.6 Section 5 Discussion

### Cambios recomendados
Agregar un parrafo corto que conecte 2 hallazgos:
1. Validez metodologica del protocolo leakage-free (Obs. 2).
2. Dependencia del desempeno respecto a augmentacion sintetica en ablacion controlada sobre benchmark aumentado (Obs. 4).

### Texto sugerido (ingles)
The additional analyses should be interpreted in two complementary blocks: (i) a leakage-free sensitivity audit confirming strict fold-level separation and test-set cleanliness (Obs. 2), and (ii) a controlled ablation on the augmented benchmark showing strong dependence on synthetic augmentation while no aggregate change was observed after removing gender (Obs. 4). Together, these findings motivate future validation on larger expert-labeled beginner cohorts.

## 3.7 Section 6 Conclusions

### Cambio recomendado
Incorporar una frase final de cierre metodologico.

### Texto sugerido (ingles)
Beyond the historical benchmark, the added analyses provide comparable-condition evidence that separately clarifies protocol integrity in leakage-free evaluation (Obs. 2) and the role of synthetic augmentation in controlled ablation on the augmented benchmark (Obs. 4), contextualizing the current contribution and limitations of the proposed system.

## 4) Cambios en tablas actuales

1. No es obligatorio modificar las tablas actuales 2-8.
2. Si deseas, puedes agregar una nota metodologica breve al pie de la tabla de resultados de clasificacion para indicar que corresponde al benchmark historico.

## 5) Cambios minimos para cumplir observaciones sin tocar figuras

Si quieres la version mas conservadora (sin regenerar ninguna figura), el minimo viable es:
1. Ajuste en Abstract (1 frase).
2. 1 parrafo en Methods (ablacion comparable + baseline simple).
3. 1 frase aclaratoria al inicio de Results 4.1.
4. Agregar tabla de leakage-free audit (obs. 2).
5. Agregar tabla de ablacion comparable (obs. 4).
6. 1 parrafo adicional en Discussion.
7. 1 frase adicional en Conclusions.

## 6) Recomendacion final

Puedes mantener todas las figuras actuales. Para responder de forma robusta a evaluadores, lo mas importante no es redibujar, sino agregar dos bloques de evidencia tabular nuevos en Results (leakage-free audit y ablation comparable) y ajustar el framing textual para separar benchmark historico vs protocolo corregido.
