# Protocolo de validacion (tesis y paper)

## 1. Proposito y alcance
Este protocolo define la validacion externa prospectiva del sistema. Busca comprobar:
- Deteccion de carencias vocales en la evaluacion inicial.
- Capacidad del sistema para reflejar mejoras tras 2 semanas de uso.

El modelo se usa solo como componente de deteccion dentro de la app. La evidencia principal del paper se basa en datos reales y etiquetas de experto.

## 2. Diseno del estudio
- Tipo: prospectivo, longitudinal, dentro de sujeto.
- Dos evaluaciones por usuario: baseline y follow_up.
- Duracion de intervencion: ~2 semanas.
- Evaluacion experta ciega a las predicciones del modelo.
- Modelo congelado antes de la recoleccion.

## 3. Hipotesis
- H1: El modelo detecta carencias similares a las del experto en baseline.
- H2: El numero de carencias detectadas disminuye en follow_up para usuarios adherentes.
- H3: El cambio detectado por el modelo se alinea con el veredicto del experto.

## 4. Participantes y reclutamiento
- N objetivo: 40 participantes.
- Reclutamiento: voluntarios con interes en entrenamiento vocal.
- Criterios de inclusion:
  - Adultos con capacidad de completar 2 semanas.
  - Acceso a dispositivo compatible.
  - Consentimiento para audio y video.
- Criterios de exclusion:
  - Patologia vocal activa.
  - Incapacidad de completar el protocolo.

## 5. Preparacion y control de condiciones
- Mismo dispositivo y microfono en baseline y follow_up (si es posible).
- Lugar con ruido ambiental bajo.
- Distancia estable al microfono.
- Instrucciones identicas en ambas evaluaciones.

## 6. Flujo del estudio
1) Baseline en la app.
2) Plan de ejercicios dentro de la app por ~2 semanas.
3) Follow_up en la app con el mismo set de tareas.
4) Revision experta de las grabaciones.

## 7. Tareas de evaluacion (recomendadas)
La evaluacion debe permitir calcular las 10 metricas usadas por el modelo:
- Notas sostenidas (long_tones/straight) para: meanRmsDb, rmsConsistency, durationSec, attackLatencyMs, stabilityCents.
- Messa di voce (long_tones/messa) para: dynamicRangeDb.
- Escalas (scales/straight) para: precisionCents.
- Escalas (scales/fast_piano) para: rangeMinMidi, rangeMaxMidi, rangeSpanSemitones.

Nota: si la app no usa exactamente estas tareas, documentar equivalencias claras.

## 8. Datos registrados por evaluacion
Tabla: Evaluation
- evaluation_id
- user_id
- session: baseline | follow_up
- timestamp
- metrics (10):
  - meanRmsDb
  - rmsConsistency
  - dynamicRangeDb
  - durationSec
  - attackLatencyMs
  - precisionCents
  - stabilityCents
  - rangeMinMidi
  - rangeMaxMidi
  - rangeSpanSemitones
- predicted_weak_groups: lista de G1..G5
- model_version
- device_info (opcional)

Tabla: Media
- media_id
- evaluation_id
- file_path
- modality: audio | video

Tabla: ExpertReview
- review_id
- evaluation_id
- expert_id
- expert_weak_groups: lista de G1..G5
- expert_overall_progress: improved | no_change | worse
- notes

## 9. Enmascaramiento (blinding)
- El experto no ve predicciones del modelo.
- El experto solo analiza las grabaciones y una rubrica estandar.

## 10. Metricas y analisis
### Deteccion (por grupo)
- Precision, recall, F1 (macro y por clase).
- Matriz de confusion por grupo.
- PR-AUC cuando existan probabilidades.

### Mejora (pre vs post)
- Cambio en numero de carencias por usuario.
- Wilcoxon signed-rank para diferencias pre/post.
- Tamano de efecto (r o Cohen d).

### Alineacion con experto
- Kappa (mejora/no mejora).
- Porcentaje de acuerdo por grupo.

## 11. Manejo de datos faltantes
- Marcar evaluaciones incompletas.
- Reportar porcentaje de abandono.
- Excluir de analisis principal si no hay baseline y follow_up.

## 12. Control de calidad
- Verificar rangos plausibles de metricas.
- Revisar outliers por usuario.
- Registrar errores de captura o fallos del API.

## 13. Etica y consentimiento
- Consentimiento informado para audio y video.
- Permitir retiro en cualquier momento.
- Almacenar datos con acceso restringido.

## 14. Entregables esperados
- Reporte estadistico de deteccion y mejora.
- Tablas y figuras listas para paper.
- Registro claro de version del modelo y fecha de congelamiento.

## 15. Checklist previo a la recoleccion
- Modelo congelado.
- Rubrica del experto definida.
- Instrucciones y tareas estandarizadas.
- Esquema de base de datos validado.
