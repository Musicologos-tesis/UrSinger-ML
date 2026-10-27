# Respuesta para la observacion 3 - Informe de ampliacion y aclaracion de metricas

Fecha: 2026-10-27

## Objetivo

Atender la solicitud del evaluador de ampliar la evaluacion para evitar interpretaciones basadas solo en exactitud global y aportar una caracterizacion mas completa del desempeno.

## Alcance de la ampliacion

La evaluacion se amplio en cinco niveles:
1. Metricas por dimension vocal: precision, recall, F1 y soporte por clase.
2. Metricas multietiqueta globales: micro-F1, macro-F1, Hamming loss y subset accuracy.
3. Variabilidad por folds: desviacion estandar e intervalo de confianza al 95%.
4. Definicion explicita de exactitud agregada.
5. Analisis focalizado de falsos negativos para identificar dimensiones mas vulnerables.

## Definicion de exactitud agregada

La exactitud agregada se define como la proporcion de decisiones binarias correctas sobre el total de decisiones binarias del problema multietiqueta, es decir, sobre todas las muestras y todas las etiquetas.

## Resultados por dimension vocal (holdout)

| Grupo | Precision | Recall | F1 | Soporte positivo | Soporte negativo | Falsos negativos |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 1.0000 | 0.5000 | 0.6667 | 10 | 60 | 5 |
| G2 | 1.0000 | 0.4000 | 0.5714 | 10 | 60 | 6 |
| G3 | 0.9000 | 0.9000 | 0.9000 | 10 | 60 | 1 |
| G4 | 0.9000 | 0.9000 | 0.9000 | 10 | 60 | 1 |
| G5 | 1.0000 | 0.9000 | 0.9474 | 10 | 60 | 1 |

## Metricas multietiqueta

### Holdout

1. micro-F1: 0.8182
2. macro-F1: 0.7971
3. micro-precision: 0.9474
4. macro-precision: 0.9600
5. micro-recall: 0.7200
6. macro-recall: 0.7200
7. Hamming loss: 0.0457
8. Subset accuracy: 0.7857
9. Exactitud agregada: 0.9543

### Validacion cruzada (promedio)

1. micro-F1: 0.8336
2. macro-F1: 0.8421
3. Hamming loss: 0.0419
4. Subset accuracy: 0.8071

## Variabilidad por fold (validacion cruzada)

Para XGBoost en 5 folds:
1. Accuracy: 0.9581 +/- 0.0169 (DE), IC95 semiancho 0.0148.
2. F1: 0.8421 +/- 0.0380 (DE), IC95 semiancho 0.0333.
3. PR-AUC: 0.9421 +/- 0.0150 (DE), IC95 semiancho 0.0131.

## Evidencia de error por omision (falsos negativos)

1. G1 y G2 concentran la mayor dificultad relativa.
2. En holdout: G1 presenta 5 FN y G2 presenta 6 FN sobre 10 positivos cada uno.
3. En validacion cruzada, G1 y G2 mantienen mayor carga de FN que G3-G5.

## Interpretacion

1. El sistema mantiene alta precision, pero la sensibilidad no es uniforme entre dimensiones.
2. G1 y G2 son las dimensiones mas desafiantes por mayor omision de casos positivos.
3. Reportar solo exactitud global ocultaba esta asimetria; las metricas ampliadas corrigen ese problema de interpretacion.

## Conclusion para la observacion 3

La observacion fue atendida con una ampliacion metodologica completa del esquema de evaluacion. El reporte ahora incluye metrica por dimension, metrica multietiqueta, variabilidad estadistica por fold y analisis de falsos negativos, ofreciendo una interpretacion mas robusta y transparente del rendimiento del sistema.
