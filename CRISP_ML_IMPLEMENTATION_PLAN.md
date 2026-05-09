# Plan de Implementación (CRISP-ML) — UrSinger ML

**Objetivo del sistema**

Se implementará un sistema de Machine Learning que, a partir de métricas vocales calculadas sobre emisiones/ejercicios, **detectará carencias** en 5 grupos de habilidad (**G1–G5**) y expondrá el resultado mediante una API para consumo por el backend.

**Alcance (scope)**

- Se entrenarán y versionarán modelos para predecir `weak_G1` … `weak_G5`.
- Se mantendrá un pipeline reproducible para: preparación de datos → entrenamiento → evaluación → empaquetado de artefactos → despliegue → monitoreo.
- Se entregará un contrato de inferencia estable (request/response) para integrarse con el backend.

**Fuera de alcance (non-goals)**

- No se implementará captura/segmentación avanzada de audio en cliente; se asumirá que el cliente entregará métricas ya calculadas (según contrato).
- No se realizará diagnóstico clínico; el sistema se limitará a detección de carencias técnicas para guiar aprendizaje.

---

## 0) Puntos de partida del repo (se tomará como base)

Se tomará como punto de partida el estado actual del proyecto:

- **Datasets** en CSV (incluyendo el aumentado) bajo `data/`.
- **Scripts** de preparación y aumento:
  - `create_skill_groups_dataset.py` (creación de labels `weak_G*`).
  - `augment_dataset.py` (aumento sintético de carencias).
- **Entrenamiento/evaluación**: `src/ursinger_ml/models/train.py` (entrenamiento de modelos XGBoost + `evaluation_report.json`).
- **Artefactos del modelo** bajo `models/`:
  - `models/saved/` (modelos `xgb_weak_G*.pkl`).
  - `models/scalers/` (`feature_scaler.pkl`, `gender_encoder.pkl`).
  - `models/reports/` (`evaluation_report.json`).
- **API**: `api/main.py` (FastAPI con endpoints de predicción y health-check).

---

## 1) Entendimiento del problema (Business / Project Understanding)

### 1.1. Definición del problema y usuarios

Se definirá formalmente:

- **Usuarios**: cantantes (usuarios finales) y el backend (consumidor de la predicción).
- **Decisión**: por cada grupo `G1…G5`, el modelo entregará:
  - una **probabilidad** de carencia (score), y
  - un **indicador binario** (carencia/no carencia) derivado de un umbral.
- **Salida de valor**: el backend traducirá carencias en recomendaciones/ejercicios (fuera del alcance de ML, pero se deberá soportar con un contrato de salida estable).

### 1.2. Métricas de éxito

Se acordarán métricas offline y online:

- **Offline (validación del modelo)**:
  - `F1` por grupo (`weak_G1…weak_G5`).
  - `Recall` por grupo (para minimizar falsos negativos en carencias importantes).
  - `Precision` por grupo (para evitar falsas alarmas excesivas).
  - `ROC-AUC` o `PR-AUC` por grupo (según desbalance).
- **Online (producto)**:
  - tasa de errores del servicio (5xx), latencia p95, y disponibilidad.
  - estabilidad de predicciones (no cambios drásticos ante inputs similares).

Se definirá un **mínimo aceptable** por grupo (por ejemplo, `F1 >= X` y `Recall >= Y`) antes de promover un modelo a producción.

### 1.3. Restricciones y supuestos

Se documentarán supuestos clave:

- El cliente enviará métricas con rangos válidos.
- Las predicciones serán **asistivas**, no médicas.
- Se mantendrá compatibilidad con el contrato del endpoint `/predict`.

### 1.4. Entregables de la fase

- Documento de requerimientos (definiciones + métricas de éxito + supuestos).
- Criterios de aceptación de “modelo listo para release”.

---

## 2) Entendimiento de datos (Data Understanding)

### 2.1. Inventario de fuentes

Se inventariarán y describirán:

- Dataset base (`data/training_data.csv`).
- Dataset con labels por grupo (`data/training_data_with_skill_groups.csv`).
- Dataset aumentado (`data/training_data_augmented.csv`).

Se registrará:

- esquema de columnas (features y targets),
- rangos esperados y unidades (dB, ms, cents, MIDI),
- cobertura por género y por cantante.

### 2.2. Análisis exploratorio

Se realizará EDA para:

- distribución de cada feature y detección de outliers,
- correlación entre features,
- distribución de clases por `weak_G*`,
- dependencia por cantante (riesgo de fuga de información si se mezcla el mismo cantante en train/test).

### 2.3. Calidad y consistencia

Se validará:

- valores faltantes y estrategias (imputación o descarte),
- consistencia entre `rangeMinMidi`, `rangeMaxMidi`, `rangeSpanSemitones`,
- validación de rangos (por ejemplo, `meanRmsDb` típico negativo).

### 2.4. Entregables de la fase

- Reporte EDA (tablas y gráficos) con hallazgos y decisiones.
- Checklist de calidad de datos (reglas de validación que se mantendrán en training e inferencia).

---

## 3) Preparación de datos (Data Preparation)

### 3.1. Definición de features y targets

Se consolidará la definición canónica:

- Features:
  - `gender`, `meanRmsDb`, `rmsConsistency`, `dynamicRangeDb`, `durationSec`, `attackLatencyMs`, `precisionCents`, `stabilityCents`, `rangeMinMidi`, `rangeMaxMidi`, `rangeSpanSemitones`.
- Targets:
  - `weak_G1…weak_G5`.

Se asegurará que **la lista de features** sea idéntica en:

- generación de dataset,
- entrenamiento,
- inferencia en API.

### 3.2. Estrategia de partición (train/test)

Se definirá una estrategia de split que reduzca fuga de información:

- Se implementará un split por **cantante** (`singer_id`) si está disponible en el dataset final, para evitar que un mismo cantante aparezca en train y test.
- Si no se dispone de `singer_id` en el dataset usado en producción, se documentará explícitamente la limitación y se propondrá incorporarlo al dataset de entrenamiento.

Se mantendrá un conjunto de test “congelado” para comparabilidad entre releases.

### 3.3. Normalización y encoding

Se estandarizará el preprocesamiento:

- `gender` se codificará con `LabelEncoder`.
- el resto de features se normalizarán con `StandardScaler`.

Se guardarán y versionarán:

- `gender_encoder.pkl`
- `feature_scaler.pkl`

### 3.4. Aumento sintético (data augmentation)

Se utilizará aumento sintético controlado:

- Se marcará explícitamente qué filas serán **sintéticas** (por ejemplo, una columna `is_synthetic`) para permitir análisis separado.
- Se evaluará sensibilidad del modelo a la proporción de datos sintéticos.
- Se definirá una política para limitar degradaciones irreales (cap de valores extremos).

### 3.5. Entregables de la fase

- Script/pipeline reproducible que genere el dataset final.
- Dataset versionado (o con hash/metadata) y reporte de distribución final.

---

## 4) Modelado (Modeling)

### 4.1. Baselines y selección de enfoque

Se establecerá un baseline simple y uno principal:

- baseline: modelos lineales o árboles simples para referencia.
- modelo principal: XGBoost por grupo (enfoque multi-label mediante 5 binarios).

### 4.2. Entrenamiento por grupo

Se entrenarán 5 modelos:

- `xgb_weak_G1.pkl`
- `xgb_weak_G2.pkl`
- `xgb_weak_G3.pkl`
- `xgb_weak_G4.pkl`
- `xgb_weak_G5.pkl`

Se parametrizará y registrará:

- hiperparámetros por modelo,
- semilla (`random_state`),
- versión del dataset,
- versión de librerías.

### 4.3. Manejo de desbalance

Se ajustará el entrenamiento para clases desbalanceadas:

- uso de `scale_pos_weight` por grupo (si aplica),
- evaluación con PR-AUC,
- ajuste de umbrales por grupo basado en objetivos de negocio (por ejemplo, priorizar recall en ciertos grupos).

### 4.4. Calibración de probabilidades

Se evaluará calibración (p. ej. Platt/Isotonic) si se requerirá que el score sea interpretable como probabilidad “real”.

### 4.5. Entregables de la fase

- Modelos y preprocesadores serializados.
- Metadata de entrenamiento (JSON) con configuración completa.

---

## 5) Evaluación (Evaluation)

### 5.1. Métricas por grupo

Se evaluará por cada `weak_G*`:

- `Accuracy`, `Precision`, `Recall`, `F1`.
- matriz de confusión.
- PR-AUC (si hay desbalance).

Se generará un reporte consolidado por release.

### 5.2. Selección de umbrales

Se seleccionarán umbrales por grupo:

- se generarán curvas precision-recall,
- se elegirá un punto de operación por objetivos (minimizar FN o FP),
- se documentarán los umbrales y el porqué.

Los umbrales se aplicarán en inferencia como parte del contrato de salida.

### 5.3. Robustez

Se verificará:

- sensibilidad a inputs extremos dentro de rango,
- consistencia (misma entrada → misma salida),
- estabilidad por género (evaluación estratificada por `F`/`M`).

### 5.4. Entregables de la fase

- Reporte de evaluación en JSON + resumen para stakeholders.
- Umbrales aprobados para producción.

---

## 6) Despliegue (Deployment)

### 6.1. Empaquetado del modelo

Se preparará un bundle de release:

- modelos `xgb_weak_G*.pkl`
- `feature_scaler.pkl`
- `gender_encoder.pkl`
- metadata de versión (por ejemplo: `model_version`, fecha, hash de dataset).

### 6.2. Contrato de inferencia

Se mantendrá un contrato estable:

- request: métricas vocales validadas.
- response: lista de carencias detectadas, scores por grupo y métricas por grupo.

Se documentará el contrato en la documentación de API.

### 6.3. Operación del servicio

Se implementará operación robusta:

- endpoint `/health` para readiness/liveness.
- logging estructurado de requests (sin datos sensibles).
- control de CORS por entorno.

### 6.4. Seguridad y límites

Se configurará:

- rate limiting (si se requiere),
- autenticación (si se requiere),
- límites de tamaño (batch) y timeouts.

### 6.5. Entregables de la fase

- Servicio FastAPI desplegable.
- Guía de despliegue (dev/prod) y variables de entorno.

---

## 7) Monitoreo y mantenimiento (Monitoring & Maintenance)

### 7.1. Monitoreo técnico (servicio)

Se instrumentará:

- tasa de errores (4xx/5xx),
- latencia p50/p95/p99,
- saturación (CPU/RAM) del contenedor/VM.

Se definirán alertas (por ejemplo, p95 > X ms o error rate > Y%).

### 7.2. Monitoreo de datos (drift)

Se monitoreará drift de features:

- cambios en distribuciones de `meanRmsDb`, `precisionCents`, etc.
- detección de inputs fuera de rango (data quality).

Se definirá una política:

- si se detectara drift significativo, se disparará un análisis y posible re-entrenamiento.

### 7.3. Monitoreo de performance (cuando existan labels reales)

Se habilitará retroalimentación (si el producto lo permite):

- recolección de labels/feedback,
- evaluación periódica en datos reales,
- comparación contra el baseline y el último release.

### 7.4. Ciclo de re-entrenamiento y versionado

Se establecerá un ciclo:

- re-entrenamiento programado (mensual/trimestral) o por drift.
- versionado semántico de modelos (por ejemplo, `vMAJOR.MINOR.PATCH`).
- rollback rápido a una versión anterior si se degradara performance.

### 7.5. Entregables de la fase

- Dashboard/indicadores de monitoreo.
- Runbook de incidentes (qué hacer ante degradación o caída del servicio).

---

## 8) Plan de ejecución (hitos)

Se organizará el trabajo en hitos verificables:

1. **Hito A — Definición y criterios**: se cerrarán métricas de éxito y umbrales objetivo.
2. **Hito B — Datos listos**: se consolidará dataset, split robusto y validaciones.
3. **Hito C — Modelos listos**: se entrenarán modelos y se generará reporte de evaluación.
4. **Hito D — API lista**: se empaquetará modelo, contrato y despliegue.
5. **Hito E — Monitoreo**: se instrumentará health/metrics/drift y runbook.

---

## 9) Riesgos y mitigaciones

- **Riesgo: fuga de información (mismo cantante en train/test)**
  - Se mitigará implementando split por `singer_id`.
- **Riesgo: sobreajuste a datos sintéticos**
  - Se mitigará etiquetando datos sintéticos y evaluando performance real vs sintético.
- **Riesgo: desbalance por grupo**
  - Se mitigará con métricas apropiadas (PR-AUC) y ajuste de umbrales/weights.
- **Riesgo: incompatibilidad training vs inferencia**
  - Se mitigará versionando y reutilizando exactamente los mismos preprocesadores.

---

## 10) Criterio de “Done” (definición de completitud)

Se considerará “implementación CRISP-ML completada” cuando:

- se habrá documentado el problema, métricas y supuestos,
- se habrá consolidado el pipeline de datos con validaciones,
- se habrán entrenado y evaluado modelos con reporte reproducible,
- se habrá desplegado la API con contrato estable y health-check,
- se habrá definido y habilitado monitoreo técnico y de drift,
- se habrá establecido un procedimiento de versionado y rollback.

---

## Anexo A) Ejecución local (comandos)

Se dejará un flujo de ejecución local repetible para que cualquier miembro del equipo pueda:

1) regenerar datasets,
2) re-entrenar,
3) evaluar,
4) levantar la API y validar predicciones.

### A.1. Preparación de entorno

Se creará y activará un entorno virtual, y se instalarán dependencias:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### A.2. Generación de datasets

Se generará el dataset con grupos y luego el dataset aumentado:

```bash
python create_skill_groups_dataset.py
python augment_dataset.py
```

Se verificará que existan:

- `data/training_data_with_skill_groups.csv`
- `data/training_data_augmented.csv`

### A.3. Entrenamiento y evaluación

Se entrenarán los 5 modelos y se guardarán artefactos y reportes:

```bash
python src/ursinger_ml/models/train.py
```

Se verificará que existan:

- `models/saved/xgb_weak_G1.pkl` … `models/saved/xgb_weak_G5.pkl`
- `models/scalers/feature_scaler.pkl`
- `models/scalers/gender_encoder.pkl`
- `models/reports/evaluation_report.json`

### A.4. Pruebas del modelo (sanity check)

Se ejecutará una verificación end-to-end con ejemplos del dataset:

```bash
python test_model.py
```

### A.5. Levantar la API

Se levantará la API para consumo del backend:

```bash
cd api
pip install -r requirements.txt

# Opción 1
python main.py

# Opción 2
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Se validará:

- `/health` responderá con `status=healthy` cuando modelos y preprocesadores estén cargados.
- `/predict` devolverá scores y carencias detectadas para entradas válidas.

