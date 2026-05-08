# Registro de cambios (explicado para principiantes)

Este documento explica claramente que se cambio o agrego en el proyecto y por que.

## 1. Cambios en entrenamiento y evaluacion
- Se agrego validacion cruzada por cantante con GroupKFold.
	- Que es: en lugar de mezclar todo al azar, se separan cantantes completos en distintos folds.
	- Por que: evita que el mismo cantante aparezca en train y test, lo que inflaba los resultados.
- Se agrego un split por cantante con GroupShuffleSplit.
	- Que es: divide train/test sin mezclar cantantes.
	- Por que: simula mejor el uso real con usuarios nuevos.
- Se agregaron modelos baseline: LogisticRegression y RandomForest.
	- Que es: modelos simples para comparar con XGBoost.
	- Por que: si XGBoost no supera a los baselines, no hay evidencia fuerte.
- Se agrego PR-AUC cuando hay probabilidades.
	- Que es: una metrica que funciona mejor cuando hay clases desbalanceadas.
	- Por que: hay pocos casos con carencias, entonces accuracy no es suficiente.
- Se agrego modo CV y opciones de entrenamiento en el script.
	- Ahora se puede usar --mode cv o --mode train.
	- Tambien se puede usar --split-strategy group o random.
- Se genero un reporte de CV: models/reports/cv_report.json.
	- Contiene resultados por fold y un resumen por modelo.

## 2. Cambios en visualizaciones
- Se agrego un grafico de resumen de CV: presentation_images/7_cv_summary.png.
	- Muestra accuracy, F1 y PR-AUC por modelo.
- Se tradujeron titulos y etiquetas a ingles para usar en paper.
- El diagrama de pipeline ahora usa estadisticas reales del dataset.
	- Muestra cantidad real de muestras y accuracy promedio actual.
- Se corrigieron las fuentes de audio para rango (fast_piano).
	- Antes aparecia low_piano, ahora coincide con el codigo real.

## 3. Cambios en pruebas de la API
- Se reescribio api/test_api.py para validar mejor la API.
	- Verifica que el response tenga todas las llaves esperadas.
	- Incluye un candidato extremo para forzar multiples carencias.
	- Incluye un barrido de muestras del dataset para ver el comportamiento general.
- Resultado observado: la API si detecta multiples carencias en casos extremos.

## 4. Dependencias agregadas
- Se agrego seaborn en requirements.txt.
	- Necesario para graficos con estilo consistente.

## 5. Archivos modificados o agregados
- src/ursinger_ml/models/train.py (entrenamiento, CV, baselines, PR-AUC)
- generate_presentation_visuals.py (graficos en ingles y CV)
- requirements.txt (seaborn)
- api/test_api.py (tests robustos de API)
- docs/validation_protocol.md, docs/changes_made.md, docs/paper_plan.md (documentacion)

## 6. Salidas actualizadas (artefactos)
- models/reports/evaluation_report.json (reporte de test)
- models/reports/cv_report.json (reporte de CV)
- models/saved/training_info.json (metadata del entrenamiento)
- models/saved/xgb_weak_G1.pkl .. xgb_weak_G5.pkl (modelos)
- models/scalers/feature_scaler.pkl (scaler actualizado)
- models/scalers/gender_encoder.pkl (encoder actualizado)
- presentation_images/*.png (graficos regenerados en ingles)

## 7. Comandos ejecutados
- py src/ursinger_ml/models/train.py --mode cv --cv-splits 5
- py src/ursinger_ml/models/train.py --mode train --split-strategy group
- py generate_presentation_visuals.py
- py -m pip install -r requirements.txt
- py -m pip install seaborn
- py api/main.py
- py api/test_api.py

## 8. Notas importantes
- La API no tiene endpoint raiz (/), por eso el test muestra 404 en /. No es un error critico.
- XGBoost muestra un warning sobre use_label_encoder, pero no afecta el resultado.
- FastAPI muestra warnings de deprecacion de Pydantic; no afecta el funcionamiento.
- En Windows se uso el launcher de Python: py.
