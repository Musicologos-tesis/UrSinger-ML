# Respuesta para la observacion 2 - Informe de auditoria de fuga de datos

Fecha: 2026-10-27

## Objetivo

Atender la observacion del evaluador sobre posible fuga de datos entre entrenamiento y prueba, y demostrar con evidencia que el protocolo fue auditado y corregido.

## Diagnostico

La auditoria identifico dos hallazgos:
1. No existia cruce de grupos de cantante entre entrenamiento y prueba.
2. Si existia contaminacion metodologica en el esquema anterior, porque el conjunto de prueba incluia muestras sinteticas generadas antes de la particion.

En terminos practicos, el problema no era overlap de grupos, sino presencia de datos sinteticos en test.

## Correccion metodologica aplicada

Se adopto un protocolo leakage-free con este orden estricto:
1. Primero se realiza la particion por grupos de cantante.
2. Luego se aplica augmentacion sintetica solo al subconjunto de entrenamiento de cada fold.
3. El subconjunto de prueba permanece sin augmentacion.

Adicionalmente, cada muestra sintetica conserva el identificador de grupo de su muestra de origen.

## Evidencia de auditoria por fold

### Esquema previo (legacy)

| Fold | Interseccion de grupos train-test | Filas sinteticas en test | Estado |
|---|---:|---:|---|
| 1 | 0 | 35 | Leak |
| 2 | 0 | 30 | Leak |
| 3 | 0 | 30 | Leak |
| 4 | 0 | 25 | Leak |
| 5 | 0 | 25 | Leak |

### Esquema corregido (leakage-free)

| Fold | Interseccion de grupos train-test | Filas sinteticas en test | Estado |
|---|---:|---:|---|
| 1 | 0 | 0 | OK |
| 2 | 0 | 0 | OK |
| 3 | 0 | 0 | OK |
| 4 | 0 | 0 | OK |
| 5 | 0 | 0 | OK |

## Verificacion de criterio de exito

El criterio de exito definido fue:
1. Interseccion de grupos train-test igual a 0 en todos los folds.
2. Filas sinteticas en test igual a 0 en todos los folds.

Ambas condiciones se cumplieron en 5/5 folds con el protocolo corregido.

## Nota para interpretar metricas despues de la correccion

En la corrida leakage-free sobre el dataset base, los folds de prueba quedaron sin positivos reales de debilidad vocal. Por esa razon pueden observarse accuracies altas junto con F1 muy bajo o nulo para la clase positiva.

Esto no invalida la correccion metodologica: confirma que la auditoria de fuga fue efectiva, pero tambien que la evaluacion de capacidad discriminativa requiere prueba con positivos reales no sinteticos.

## Conclusion para la observacion 2

La observacion fue subsanada. El protocolo actual evita fuga de datos al separar por grupos antes de la augmentacion y restringir la sintesis al entrenamiento. La auditoria por fold confirma ausencia simultanea de overlap de grupos y de muestras sinteticas en test.

