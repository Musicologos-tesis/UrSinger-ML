# Respuesta para la observacion 4 - Informe de comparaciones y ablacion

Fecha: 2026-10-27

## Objetivo

Para atender la observacion del evaluador, se realizo un estudio comparativo con tres focos:
1. Comparar el modelo propuesto con un clasificador simple por reglas/umbrales.
2. Comparar entrenamiento con y sin aumento sintetico.
3. Evaluar el efecto de retirar la variable de genero.

## Diseno de evaluacion

Se aplico un protocolo unico y comparable para todos los escenarios:
1. Validacion cruzada en 5 folds.
2. Particion por grupos de cantante para evitar mezcla de identidad entre entrenamiento y prueba.
3. Mismas particiones para todos los escenarios (comparacion controlada).
4. Mismas metricas de evaluacion en todos los casos.

Importante para interpretacion:
- Esta ablacion se ejecuto sobre el benchmark aumentado/sintetico de desarrollo.
- Por diseno de este benchmark, los folds de prueba pueden incluir muestras sinteticas.
- Por tanto, este bloque debe interpretarse como comparacion controlada de escenarios, no como auditoria leakage-free.

## Escenarios evaluados

1. Modelo propuesto (XGBoost) con datos sinteticos y con genero.
2. XGBoost sin datos sinteticos y con genero.
3. XGBoost con datos sinteticos y sin genero.
4. XGBoost sin datos sinteticos y sin genero.
5. Heuristic rule-threshold baseline.

## Definicion del heuristic rule-threshold baseline

El baseline se construyo como una referencia interpretable por reglas/umbrales:
1. G1: debilidad si duracion < 1.2 s.
2. G2: debilidad si error de afinacion > 50 cents.
3. G3: debilidad si inestabilidad de afinacion > 50 cents.
4. G4: debilidad si rango dinamico < 10 dB.
5. G5: debilidad si extension vocal < 6 semitonos.

Estos umbrales se usan solo como referencia comparativa interpretable y no como thresholds clinicos ni diagnosticos.

## Resultados (promedio en 5 folds)

| Escenario | Accuracy | F1 | Micro-F1 | Macro-F1 | Hamming loss | Subset accuracy |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost + sintetico + genero | 0.9581 | 0.8421 | 0.8336 | 0.8421 | 0.0419 | 0.8071 |
| XGBoost - sintetico + genero | 0.8818 | 0.0000 | 0.0000 | 0.0000 | 0.1182 | 0.4089 |
| XGBoost + sintetico - genero | 0.9581 | 0.8421 | 0.8336 | 0.8421 | 0.0419 | 0.8071 |
| XGBoost - sintetico - genero | 0.8818 | 0.0000 | 0.0000 | 0.0000 | 0.1182 | 0.4089 |
| Heuristic rule-threshold baseline | 0.9196 | 0.4262 | 0.4850 | 0.4262 | 0.0804 | 0.5982 |

## Transparencia de composicion del test en ablacion

En esta ablacion controlada (benchmark aumentado), la composicion del test por fold incluye originales y sinteticos.

| Fold | Filas originales en test | Filas sinteticas en test | Proporcion sintetica |
|---|---:|---:|---:|
| 1 | 15 | 35 | 0.70 |
| 2 | 20 | 30 | 0.60 |
| 3 | 20 | 30 | 0.60 |
| 4 | 25 | 25 | 0.50 |
| 5 | 20 | 25 | 0.56 |

Esta trazabilidad confirma que la ablacion de esta observacion no corresponde al protocolo leakage-free de la observacion 2.

## Evidencia cuantitativa clave

1. Aporte de aumento sintetico (con genero):
- F1: +0.8421 (de 0.0000 a 0.8421)
- Micro-F1: +0.8336 (de 0.0000 a 0.8336)
- Subset accuracy: +0.3982 (de 0.4089 a 0.8071)
- Hamming loss: -0.0763 (de 0.1182 a 0.0419, menor es mejor)

2. Modelo propuesto vs baseline simple:
- F1: +0.4159 (de 0.4262 a 0.8421)
- Micro-F1: +0.3486 (de 0.4850 a 0.8336)
- Subset accuracy: +0.2089 (de 0.5982 a 0.8071)
- Hamming loss: -0.0385 (de 0.0804 a 0.0419, menor es mejor)

3. Retiro de genero (con sinteticos):
- No se observaron cambios en las metricas agregadas reportadas.

## Interpretacion

1. Comparacion con clasificador sencillo:
El modelo propuesto supera al baseline por reglas en todas las metricas multilabel clave, especialmente en F1 y subset accuracy.

2. Entrenamiento con y sin sinteticos:
La remocion del aumento sintetico reduce de forma drastica la capacidad de deteccion de positivos (F1 y micro/macro-F1 cercanos a cero), aunque la accuracy se mantiene relativamente alta por el desbalance de clases.

3. Efecto de retirar genero:
En este conjunto de datos, retirar genero no modifico el rendimiento global agregado.

4. Condiciones comparables:
La comparacion se realizo bajo particiones identicas entre escenarios, por lo que las diferencias observadas se atribuyen a los factores evaluados y no a cambios en el protocolo de prueba.

## Conclusion para la observacion 4

Se atendio la solicitud del evaluador con un analisis controlado y comparable. La evidencia muestra que:
1. El modelo propuesto aporta mejora clara frente a una alternativa simple por umbrales.
2. El aumento sintetico es el componente con mayor impacto en el desempeno de deteccion.
3. La variable de genero no mostro impacto agregado en este experimento.
4. Los resultados deben leerse en el contexto de benchmark aumentado/sintetico.

Estos resultados se reportan de forma explicita, incluyendo los escenarios no favorables al modelo, tal como fue solicitado.
