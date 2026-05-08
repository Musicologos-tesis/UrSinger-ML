# Guia detallada del paper (journal o conference)

Este documento explica con detalle como redactar el paper con enfoque en sistema completo, incluyendo el componente ML, la arquitectura web (Angular + Nest.js), y la validacion prospectiva con usuarios reales.

## 1. Titulos recomendados (elige uno)
1) "Aplicacion de entrenamiento vocal con deteccion de carencias: sistema end-to-end y validacion prospectiva"
2) "Deteccion de carencias vocales en una app de entrenamiento: arquitectura, modelo y estudio piloto"
3) "Entrenamiento vocal personalizado con metricas acusticas: sistema completo y validacion con usuarios"

## 2. Enfoque y mensaje central
- Este paper es de sistema completo, no solo de ML.
- El aporte es la integracion: front (CREPE), backend (orquestacion), API ML y plan de entrenamiento personalizado.
- El entrenamiento es weakly-supervised (etiquetas sinteticas), pero la evidencia principal es la validacion con usuarios y experto.
- El tono es pilot / proof-of-concept con resultados reales y limitaciones claras.

## 3. Contribuciones principales (para abstract e introduccion)
1) Pipeline end-to-end para evaluar voz, detectar carencias y generar un plan personalizado dentro de una app.
2) Extraccion en tiempo real de 10 metricas vocales con CREPE y logica de post-procesamiento.
3) Modelo multi-etiqueta basado en 5 clasificadores binarios (G1..G5).
4) Validacion prospectiva con usuarios reales y evaluacion experta ciega.

## 4. Mensaje de valor tecnico
El paper debe enfatizar que el sistema convierte audio crudo en decisiones pedagogicas (plan de entrenamiento), y valida que esas decisiones se alinean con expertos y con la mejora del usuario.

## 5. Estructura del paper (seccion por seccion, con detalle)

### 5.1 Abstract
Estructura sugerida (4-5 frases):
1) Problema y contexto: por que se necesita evaluacion vocal accesible.
2) Sistema propuesto: app + API + plan personalizado.
3) Modelo y metricas usadas (10 metricas, 5 grupos).
4) Validacion: N=40, 2 semanas, experto ciego.
5) Resultados clave (placeholder hasta tener datos).

### 5.2 Introduccion
- Contexto: dificultad de evaluar carencias vocales en entornos no profesionales.
- Brecha: la mayoria de apps no detecta carencias ni valida mejora real.
- Propuesta: sistema completo que detecta carencias y adapta entrenamiento.
- Lista de contribuciones (las 4 del apartado 3).
- Cierre: resumen de la estructura del paper.

### 5.3 Trabajo relacionado
- Evaluacion vocal automatizada y metricas acusticas.
- ML aplicado a voz (afinacion, estabilidad, rango).
- Apps de coaching y estudios longitudinales.

### 5.4 Arquitectura del sistema (seccion clave)
Incluye un diagrama propio del flujo:
- Frontend (Angular): captura audio en tiempo real, ejecuta CREPE y calcula metricas.
- Backend (Nest.js): recibe metricas, consulta API del modelo, genera plan de entrenamiento.
- API ML: responde grupos con carencias G1..G5.
- UI de entrenamiento: muestra ruta de ejercicios y permite practicar dentro de la app.

Datos a describir:
- Frecuencia de muestreo.
- Ventanas de analisis y smoothing.
- Mecanismo de validacion de metricas (rangos, outliers).
- Latencia end-to-end (captura -> plan).

### 5.5 Extraccion de metricas
- Enumerar las 10 metricas.
- Explicar como se calculan (resumen breve).
- Relacion con grupos G1..G5.
- Figura: presentation_images/6_metrics_extraction.png.

### 5.6 Modelo y entrenamiento
- 5 clasificadores binarios (uno por grupo).
- XGBoost como modelo principal, baselines LogisticRegression y RandomForest.
- Split por cantante y CV con GroupKFold.
- Metricas internas: F1 macro y PR-AUC.
- Figura: presentation_images/7_cv_summary.png.

### 5.7 Protocolo de validacion prospectiva
- 40 usuarios, baseline y follow_up tras 2 semanas.
- Uso real de la app durante el periodo.
- Etiquetas por experto ciego.
- Endpoints: deteccion y mejora.
- Tabla de protocolo (resumen de participantes, tareas y metrics).

### 5.8 Resultados (como se reportan)
Resultados deben mostrar tres capas:

**A) Resultados del sistema (principal)**
- Cambio pre/post en numero de carencias por usuario.
- Alineacion con experto sobre mejora (kappa).
- Ejemplo de usuario con plan y progreso (cualitativo).

**B) Resultados de deteccion (comparado con experto)**
- Precision, recall, F1 por grupo.
- Matrices de confusion por grupo.

**C) Resultados offline (soporte tecnico)**
- CV por cantante con baselines.
- Tabla comparativa de modelos.

### 5.9 Discusion
- Interpretacion de mejoras y deteccion.
- Diferencias entre datos sinteticos y reales.
- Impacto de la integracion end-to-end.
- Implicaciones para entrenamiento vocal remoto.

### 5.10 Limitaciones
- Entrenamiento con etiquetas sinteticas.
- Tamaño de muestra y periodo corto.
- Variabilidad de hardware/entorno.
- Posible sesgo por adherencia al plan.

### 5.11 Conclusion y trabajo futuro
- Resumen de aportes y resultados.
- Proximos pasos: datos reales mas grandes, mejoras de etiquetas, personalizacion mas profunda.

## 6. Tablas y figuras recomendadas
Tablas:
- T1: Definicion de metricas (10) y fuente de audio.
- T2: Resumen de CV por modelo (accuracy, F1, PR-AUC).
- T3: Validacion real (por grupo y global).
- T4: Resultados pre/post (promedio, mediana, p-valor, efecto).

Figuras:
- F1: Pipeline del sistema (1_pipeline.png).
- F2: Grupos y metricas (2_metrics_groups.png).
- F3: CV summary (7_cv_summary.png).
- F4: Matrices de confusion (4_confusion_matrices.png).
- F5: Cambio pre/post por usuario (boxplot o line plot).

## 7. Analisis estadistico (cuando haya datos reales)
- Por grupo: precision, recall, F1, PR-AUC.
- Concordancia con experto: kappa.
- Mejora pre/post: Wilcoxon y tamano de efecto.
- Reportar media, mediana y desviacion por usuario.
- Reportar intervalos de confianza (bootstrap).

## 8. Como redactar la seccion de resultados
- Empieza con un resumen claro de N, adherencia y completitud.
- Presenta primero resultados del sistema (mejora).
- Luego deteccion vs experto.
- Cierra con CV offline como evidencia tecnica adicional.
- Evita claims clinicos; usa lenguaje de evidencia preliminar.

## 9. Reglas de redaccion
- Usar "pilot" o "proof-of-concept".
- No afirmar generalizacion fuerte.
- Separar entrenamiento sintetico vs validacion real.
- Declarar el modelo como congelado durante la validacion.

## 10. Checklist antes de envio
- Modelo congelado y versionado.
- Protocolo aplicado y documentado.
- Resultados reales completos con estadistica.
- Limitaciones explicitas.
- Figuras con etiquetas en ingles.

## 11. Plantillas de texto (para completar al final)
- "We conducted a prospective validation with N=40 users over two weeks and expert-blinded labels."
- "The system reduced the average number of detected weaknesses from X to Y (p=..., effect=...)."
- "Expert agreement (kappa) at baseline was ... and improved to ... at follow-up."
