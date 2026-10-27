# Cambios posteriores a revision 2 - Guia tecnica interna

Fecha: 2026-10-27

## 1) Objetivo de estos cambios

Subsanar la observacion del revisor sobre posible fuga de datos entre train y test, asegurando que:
1. El split por grupos se haga antes de cualquier augmentacion.
2. La augmentacion ocurra solo en train.
3. Test permanezca sin muestras sinteticas.
4. No haya group_id compartido entre train y test.

## 2) Que se cambio en codigo (detalle tecnico)

### 2.1 Definicion explicita de grupo para split

Que se implemento:
1. Si no existe group_id, se crea usando singer_id.

Donde:
1. [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L93)
2. [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L253)

Impacto:
1. Todas las muestras de un mismo cantante quedan en el mismo fold.

### 2.2 Augmentacion movida al interior del train de cada fold

Que se implemento:
1. Funcion dedicada de augmentacion intra-fold.
2. Los sinteticos se generan despues del split, solo sobre train.
3. Cada sintetico hereda group_id de su muestra origen.

Donde:
1. [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L182)
2. [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L205)
3. [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L479)

Impacto:
1. Se elimina contaminacion por sinteticos en test.

### 2.3 Auditoria automatica de fuga por fold

Que se implemento:
1. CV con chequeo de interseccion de grupos train-test.
2. Conteo de filas sinteticas en test por fold.
3. Estado OK/Leak por fold.
4. Export de reporte JSON.

Donde:
1. [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L464)
2. [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L494)
3. [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L503)
4. [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L541)

Reporte generado:
1. [models/reports/leakage_audit_report.json](../models/reports/leakage_audit_report.json)

### 2.4 Soporte de reproducibilidad (modos legacy y corregido)

Que se implemento:
1. Argumentos CLI para elegir dataset y activar modo legacy.

Donde:
1. [src/ursinger_ml/models/train.py](../src/ursinger_ml/models/train.py#L684)

Impacto:
1. Permite reproducir resultados historicos y compararlos con el protocolo corregido.

## 3) Evidencia de que se subsano la observacion

### 3.1 Resultado de auditoria corregida

Archivo:
1. [models/reports/leakage_audit_report.json](../models/reports/leakage_audit_report.json)

Conclusiones:
1. Interseccion de grupos train-test = 0 en todos los folds.
2. Filas sinteticas en test = 0 en todos los folds.

### 3.2 Comparacion contra pipeline legacy

Archivo:
1. [models/reports/leakage_audit_legacy_global_augmented.json](../models/reports/leakage_audit_legacy_global_augmented.json)

Conclusiones:
1. En legacy no habia cruce de grupos, pero si habia sinteticos en test.
2. El problema metodologico era contaminacion de test, no group overlap.

## 4) Nota tecnica importante para interpretar metricas

1. El dataset base usado en el protocolo corregido inicializa weak_G1..weak_G5 en 0.
2. Por eso el test corregido puede quedar sin positivos y mostrar accuracy alta con F1 bajo/cero.

Referencia:
1. [create_skill_groups_dataset.py](../create_skill_groups_dataset.py#L34)

Implicacion:
1. La correccion valida el protocolo anti-fuga.
2. No implica automaticamente mejora de performance predictiva en ese dataset base.

## 5) Recomendacion para documentar resultados (despues del cambio de codigo)

Decision recomendada:
1. Mantener Figures 7-9 legacy como benchmark historico de comparacion.
2. Agregar en Results una subseccion nueva de sensibilidad leakage-free.
3. Agregar nota en captions para evitar observacion metodologica.

Por que:
1. Las figuras legacy son las que sostienen el mensaje de rendimiento comparativo.
2. El protocolo corregido sostiene el mensaje metodologico de ausencia de fuga.

## 6) Texto tecnico breve reutilizable (interno)

Following the leakage audit, we enforced a leakage-free protocol where group-based splitting is executed before augmentation, and synthetic data are generated only inside each training fold. This removed synthetic samples from test partitions and preserved disjoint singer groups between train and test in all folds. The updated evaluation should be interpreted as protocol-validity evidence first (leakage control), and as performance evidence only when test partitions contain real positive weakness labels.

## 7) Checklist final para tu companero

1. Verifico que en Results 4.1-4.3 se mencione explicitamente "historical augmented benchmark".
2. Agrego subseccion 4.4 completa.
3. Agrego nota de caption en Figures 7-9.
4. Agrego parrafo de limitacion en Discussion.
5. Confirmo que no haya frases que presenten Figures 7-9 como leakage-free final.
6. Confirmo que Methods describa split por grupos y augmentacion solo en train.
