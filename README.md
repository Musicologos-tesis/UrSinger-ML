# 🎤 UrSinger ML - Detección de Carencias Vocales

Sistema de Machine Learning para detectar carencias vocales en cantantes usando XGBoost.

## 📋 Descripción

Este proyecto utiliza modelos de Machine Learning para analizar métricas vocales y detectar carencias en 5 grupos de habilidad:

- **G1**: Soporte Respiratorio y Control de Aire
- **G2**: Afinación y Oído Tonal
- **G3**: Estabilidad y Control
- **G4**: Potencia y Control Dinámico
- **G5**: Rango y Flexibilidad Vocal

## 🏗️ Estructura del Proyecto

```
ml/
├── api/                        # API REST con FastAPI
│   ├── main.py                # Servicio ML con Swagger
│   ├── requirements.txt       # Dependencias de la API
│   ├── test_api.py           # Tests de la API
│   └── README.md             # Documentación de la API
├── src/                       # Código fuente
│   └── ursinger_ml/
│       ├── data/             # Procesamiento de datos
│       ├── features/         # Extracción de features
│       ├── models/           # Entrenamiento de modelos
│       └── preprocessing/    # Preprocesamiento
├── models/                    # Modelos entrenados
│   ├── saved/                # Modelos XGBoost (.pkl)
│   ├── scalers/              # Normalizadores
│   └── reports/              # Reportes de evaluación
├── data/                      # Datos (NO incluidos en repo)
│   ├── training_data_augmented.csv
│   ├── training_data_with_skill_groups.csv
│   └── vocalset/             # Dataset VocalSet (2.6 GB)
└── presentation_images/       # Visualizaciones para PPT
```

## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <tu-repo-url>
cd ml
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Obtener el dataset VocalSet

⚠️ **IMPORTANTE**: El dataset VocalSet (2.6 GB) NO está incluido en el repositorio.

**Opción 1: Descargar VocalSet**
1. Visita: https://zenodo.org/record/1442513
2. Descarga el dataset
3. Extrae en: `data/vocalset/`

**Opción 2: Usar solo los CSVs procesados**
Los archivos CSV con las métricas ya extraídas SÍ están incluidos:
- `data/training_data_augmented.csv` (245 muestras)
- `data/training_data_with_skill_groups.csv` (100 muestras)

## 🎯 Uso

### Iniciar la API ML

```bash
cd api
python main.py
```

La API estará disponible en:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API**: http://localhost:8000

### Probar la API

```bash
python api/test_api.py
```

### Entrenar el modelo (si tienes el dataset)

```bash
python src/ursinger_ml/models/train.py
```

### Generar visualizaciones

```bash
python generate_presentation_visuals.py
```

## 📊 Resultados del Modelo

- **Accuracy Promedio**: 95.5%
- **F1-Score Promedio**: 80.4%

| Grupo | Accuracy | F1-Score | Descripción |
|-------|----------|----------|-------------|
| G1 | 93.9% | 66.7% | Soporte Respiratorio |
| G2 | 98.0% | 88.9% | Afinación y Oído Tonal |
| G3 | 95.9% | 80.0% | Estabilidad y Control |
| G4 | 89.8% | 66.7% | Potencia y Control Dinámico |
| G5 | 100.0% | 100.0% | Rango y Flexibilidad Vocal |

## 🔗 Integración con Backend NestJS

Ver documentación completa en: `api/nestjs-integration-example.ts`

```typescript
// Ejemplo básico
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    gender: "F",
    meanRmsDb: -25.5,
    rmsConsistency: 3.2,
    // ... otras métricas
  })
});
```

## 📚 Documentación Adicional

- [API Documentation](api/README.md)
- [Swagger Guide](api/SWAGGER_GUIDE.md)
- [Architecture Decision](DECISION_FINAL_ARQUITECTURA.md)
- [Thesis Summary](RESUMEN_TESIS.md)

## 🛠️ Tecnologías

- **Python 3.11+**
- **FastAPI** - API REST
- **XGBoost** - Modelo de ML
- **Scikit-learn** - Preprocesamiento
- **Librosa** - Procesamiento de audio
- **CREPE** - Detección de pitch
- **Pydantic** - Validación de datos

## 👨‍💻 Autores

- Anthony - [UrSinger Team]

## 📄 Licencia

MIT License

## 🎓 Proyecto de Tesis

Este proyecto es parte de una tesis sobre detección automática de carencias vocales en cantantes usando Machine Learning.

