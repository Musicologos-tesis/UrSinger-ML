# Respuestas a observaciones adicionales

Fecha: 2026-11-02

## R1 - 2. Comparacion con umbrales heurisiticos

### Respuesta propuesta

Se aclaro el papel del baseline heuristico basado en reglas y umbrales dentro del protocolo experimental. Este baseline se incluyo como una referencia interpretable para compararlo con el enfoque propuesto basado en XGBoost. No fue diseñado para reconstruir, reproducir ni validar el procedimiento utilizado para generar las etiquetas sinteticas.

El baseline heuristico aplica umbrales operativos fijos a determinados descriptores acusticos asociados con las cinco dimensiones pedagogicas: duracion sostenida inferior a 1.2 s para G1, error de afinacion superior a 50 cents para G2, inestabilidad de afinacion superior a 50 cents para G3, rango dinamico inferior a 10 dB para G4 y extension vocal inferior a 6 semitonos para G5. Estos umbrales se incorporaron exclusivamente como reglas sencillas de comparacion. No fueron derivados de anotaciones de expertos, criterios clinicos, mediciones fisiologicas ni formulas de generacion de etiquetas sinteticas.

En consecuencia, la comparacion responde una pregunta limitada pero clara: si los clasificadores XGBoost aprendidos proporcionan un mapeo predictivo mas eficaz, dentro del benchmark aumentado de desarrollo, que una heuristica transparente sin aprendizaje automatico. El resultado no debe interpretarse como evidencia de que los umbrales heurisiticos reproduzcan el proceso de construccion de los datos sinteticos ni como validacion clinica, diagnostica o pedagogica de dichos umbrales.

### Evidencia que debe quedar en el texto

- El baseline se denomina **heuristic rule-threshold baseline**.
- Es una referencia interpretable no basada en aprendizaje automatico.
- Sus umbrales son reglas operativas fijas.
- No reconstruye las perturbaciones sinteticas de la Tabla 5.
- No valida las etiquetas sinteticas.
- No representa thresholds clinicos, diagnosticos ni fisiologicos.

### Ubicacion recomendada

Agregar esta aclaracion en la seccion **3.5. Experimental Protocol and Baselines**, inmediatamente despues de describir los cinco escenarios de la ablacion. Tambien conviene conservar una nota breve en la interpretacion de la Tabla 10.

---

## R1 - 4. Tabla 8: identificacion de F1-score y variabilidad

### Decision

**Si se debe corregir la Tabla 8.**

### Por que

La tabla actual presenta `F1-score = 0.8421 ± 0.0380`, pero no indica explicitamente que ese F1 corresponde al **Macro-F1 promedio entre los cinco targets G1-G5**, calculado en cada fold y posteriormente resumido entre los cinco folds.

Ademas, la nota de la tabla indica que Accuracy, F1-score y PR-AUC se reportan como media ± desviacion estandar, pero la tabla no deja suficientemente claro que:
1. La media y la desviacion estandar se calcularon sobre los cinco folds.
2. F1-score corresponde a macro-F1, no a micro-F1.
3. Micro-F1 y Macro-F1 aparecen tambien como metricas multilabel separadas, lo que puede generar ambiguedad si la columna general se llama simplemente F1-score.

La correccion es necesaria porque la observacion R1-4 solicita una definicion estadistica precisa y porque el lector podria interpretar el F1 general como micro-F1, weighted-F1 o una metrica agregada distinta.

### Correccion recomendada para la Tabla 8

Cambiar la fila:

```text
F1-score | 0.8421 ± 0.0380
```

por:

```text
Macro-F1 por fold (media ± DE entre folds) | 0.8421 ± 0.0380
```

Y cambiar la nota por:

```text
Nota. Accuracy, Macro-F1 por fold y PR-AUC se reportan como media ± desviacion estandar entre los cinco folds de GroupKFold. El Macro-F1 se calcula como la media aritmetica de los F1 de G1-G5 dentro de cada fold y posteriormente se resume entre folds. Micro-F1, Macro-F1, Hamming loss y subset accuracy se reportan como metricas multietiqueta promediadas entre folds cuando corresponde. Estos resultados corresponden al benchmark aumentado de desarrollo y no al protocolo leakage-free descrito en la Seccion 4.2.
```

### Observacion tecnica adicional

Conviene evitar usar simultaneamente `F1-score` como nombre general y `Macro-F1` como metrica multietiqueta en la misma tabla sin una definicion. La opcion mas clara es conservar ambas, pero con nombres explicitos:

| Metrica | Valor |
|---|---:|
| Accuracy (media ± DE entre folds) | 0.9581 ± 0.0169 |
| Macro-F1 por fold (media ± DE entre folds) | 0.8421 ± 0.0380 |
| PR-AUC (media ± DE entre folds) | 0.9421 ± 0.0150 |
| Micro-F1 multietiqueta (media entre folds) | 0.8336 |
| Macro-F1 multietiqueta (media entre folds) | 0.8421 |
| Hamming loss (media entre folds) | 0.0419 |
| Subset accuracy (media entre folds) | 0.8071 |

Si se conserva una sola fila de Macro-F1, debe indicarse expresamente si representa el promedio por target dentro de cada fold o la metrica multilabel calculada sobre todas las etiquetas. En este caso concreto, ambos valores reportados son iguales, pero la definicion debe permanecer explicita para evitar ambiguedad metodologica.

---

## R3 - 3. Descriptores acusticos como indicadores indirectos

### Respuesta propuesta

Se aclaro la interpretacion de los descriptores acusticos utilizados en el mapeo pedagogico. Variables como el nivel RMS, la consistencia RMS, el rango dinamico, la duracion de la emision sostenida, la desviacion de afinacion, la variabilidad de afinacion, la latencia de ataque y la extension vocal observada son indicadores acusticos derivados de la señal. No constituyen mediciones fisiologicas directas de la respiracion, el flujo de aire, el volumen pulmonar, la actividad de los musculos respiratorios, la funcion glotal, el comportamiento de los pliegues vocales ni otros mecanismos fisiologicos subyacentes.

En particular, los descriptores relacionados con RMS y la duracion sostenida pueden aportar informacion indirecta sobre el comportamiento acustico de una emision vocal, pero no pueden medir de forma independiente el apoyo respiratorio ni la funcion respiratoria. Del mismo modo, los descriptores relacionados con la afinacion, como la desviacion y la variabilidad de afinacion, pueden describir regularidad acustica y comportamiento relacionado con la entonacion, pero no deben interpretarse como mediciones directas de la fisiologia de los pliegues vocales ni del control vocal completo.

En consecuencia, los cinco grupos G1-G5 deben entenderse como categorias pedagogicamente motivadas y apoyadas por indicadores acusticos indirectos. El mapeo tiene como finalidad organizar comportamientos vocales medibles en dimensiones de feedback interpretables para la plataforma. No establece una interpretacion fisiologica uno a uno de ningun descriptor individual ni de ninguna salida del clasificador.

### Cambios recomendados en texto

En la seccion **3.2. Acoustic Feature Extraction and Pedagogical Modeling**, mantener o agregar el siguiente parrafo:

```text
Los descriptores acusticos utilizados en este estudio deben interpretarse como indicadores pedagogicos indirectos y no como mediciones fisiologicas directas. Las variables relacionadas con RMS, el rango dinamico, la duracion de la emision sostenida, la desviacion de afinacion, la variabilidad de afinacion, la latencia de ataque y la extension vocal observada describen propiedades medibles de la señal acustica registrada. No miden directamente el flujo de aire, el volumen pulmonar, la actividad de los musculos respiratorios, la resistencia glotal, la funcion de los pliegues vocales ni otros mecanismos fisiologicos. Por tanto, los cinco grupos de habilidades vocales representan categorias pedagogicamente motivadas cuya interpretacion depende del comportamiento conjunto de multiples descriptores y no deben tratarse como diagnosticos clinicos o fisiologicos.
```

En la seccion **3.4. Multi-Label Classification and Adaptive Feedback**, agregar una aclaracion breve:

```text
Los grupos de habilidades predichos deben interpretarse igualmente como posibles patrones acusticos asociados con dimensiones pedagogicas, y no como evaluaciones fisiologicas o clinicas directas del cantante.
```

En la seccion **5. Discussion**, reforzar la limitacion con:

```text
Debido a que los descriptores implementados son indicadores acusticos indirectos, los grupos detectados deben utilizarse para apoyar el feedback pedagogico y las recomendaciones de practica, no para inferir directamente la fisiologia respiratoria, laringea o de los pliegues vocales.
```

### Cambios recomendados en tablas

La Tabla 3 puede conservarse porque describe las variables acusticas. Sin embargo, conviene agregar una nota al pie:

```text
Nota. Todas las variables enumeradas son descriptores acusticos derivados de la señal. Se utilizan como indicadores indirectos dentro del mapeo pedagogico y no deben interpretarse como mediciones fisiologicas, clinicas o diagnosticas directas.
```

La Tabla 4 no necesita cambiar su estructura. Sus columnas de `Interpretive limitations` ya son adecuadas, pero deben conservarse y aplicarse de manera consistente a toda la narrativa del paper. En particular:
1. G1 no debe describirse como medicion directa de breath support.
2. G3 no debe describirse como medicion directa de vocal stability fisiologica.
3. G4 no debe describirse como medicion calibrada de sound pressure level o power fisiologico.
4. G5 no debe describirse como medicion completa de vocal flexibility.

---

## Resumen de decisiones

| Observacion | Decision | Accion |
|---|---|---|
| R1-2 | Aceptar y aclarar | Describir el baseline como referencia heuristica interpretable, no como reconstruccion ni validacion de etiquetas sinteticas. |
| R1-4 | Corregir | Renombrar F1-score como Macro-F1 per-fold y especificar media ± desviacion estandar entre cinco folds. |
| R3-3 | Aceptar y aclarar | Presentar RMS, pitch y demas descriptores como indicadores acusticos pedagogicos indirectos, no mediciones fisiologicas directas. |
