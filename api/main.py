"""
API REST del Modelo de Detección de Carencias Vocales
Servicio FastAPI para ser consumido por el backend NestJS
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI con documentación Swagger completa
app = FastAPI(
    title="🎤 UrSinger ML API",
    openapi_tags=[
        {
            "name": "Predicción",
            "description": "Endpoints para detectar carencias vocales"
        },
        {
            "name": "Salud",
            "description": "Endpoints para verificar el estado del servicio"
        }
    ]
)

# CORS - Configurar para que SOLO tu backend pueda acceder
# En producción, reemplaza "*" con la URL de tu backend NestJS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:4200"],  # URLs de tu backend/frontend
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Rutas de los modelos
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models" / "saved"
SCALERS_DIR = BASE_DIR / "models" / "scalers"

# Cargar modelos y scalers al iniciar la API
models = {}
scaler = None
gender_encoder = None

# Umbrales por grupo para la respuesta avanzada de /predict
GROUP_THRESHOLDS = {
    "weak_G1": 0.40,
    "weak_G2": 0.45,
    "weak_G3": 0.42,
    "weak_G4": 0.50,
    "weak_G5": 0.47,
}

# Reglas físicas de respaldo: valores tan extremos que son carencia independientemente del modelo
PHYSICAL_WEAKNESS_RULES = {
    "weak_G1": lambda m: m.durationSec < 1.2,
    "weak_G2": lambda m: m.precisionCents > 100,
    "weak_G3": lambda m: m.stabilityCents > 50,
    "weak_G4": lambda m: m.dynamicRangeDb < 10.0,
    "weak_G5": lambda m: m.rangeSpanSemitones < 6,
}

@app.on_event("startup")
async def load_models():
    """Carga los modelos y scalers al iniciar la API"""
    global models, scaler, gender_encoder

    try:
        logger.info("Cargando modelos y scalers...")

        # Cargar los 5 modelos XGBoost
        for i in range(1, 6):
            model_path = MODELS_DIR / f"xgb_weak_G{i}.pkl"
            models[f"weak_G{i}"] = joblib.load(model_path)
            logger.info(f"✓ Modelo weak_G{i} cargado")

        # Cargar scalers
        scaler = joblib.load(SCALERS_DIR / "feature_scaler.pkl")
        gender_encoder = joblib.load(SCALERS_DIR / "gender_encoder.pkl")

        logger.info("✓ Todos los modelos y scalers cargados exitosamente")

    except Exception as e:
        logger.error(f"Error cargando modelos: {e}")
        raise


# ============================================================================
# MODELOS DE DATOS (Pydantic)
# ============================================================================

class VocalMetrics(BaseModel):
    """
    Métricas vocales extraídas del audio del cantante.

    Estas métricas son calculadas por el frontend usando Web Audio API,
    librosa.js o procesamiento de audio similar.
    """
    gender: str = Field(
        ...,
        description="Género del cantante",
        example="F",
        pattern="^(F|M)$"
    )
    meanRmsDb: float = Field(
        ...,
        description="Nivel promedio de volumen durante la emisión (dBFS). Valores típicos: -35 a -15",
        example=-25.5,
        ge=-60,
        le=0
    )
    rmsConsistency: float = Field(
        ...,
        description="Estabilidad del volumen (desviación estándar en dBFS). Menor = más estable. Valores típicos: 2-8",
        example=3.2,
        ge=0
    )
    dynamicRangeDb: float = Field(
        ...,
        description="Diferencia entre volumen máximo y mínimo (dBFS). Valores típicos: 15-80",
        example=68.5,
        ge=0,
        le=120
    )
    durationSec: float = Field(
        ...,
        description="Duración efectiva de notas sostenidas (segundos). Valores típicos: 1.5-4",
        example=2.8,
        ge=0,
        le=10
    )
    attackLatencyMs: float = Field(
        ...,
        description="Tiempo entre inicio de sonido y estabilización del tono (milisegundos). Valores típicos: 50-150",
        example=85.3,
        ge=0,
        le=500
    )
    precisionCents: float = Field(
        ...,
        description="Diferencia promedio entre pitch emitido y objetivo (cents). Menor = mejor. Valores típicos: 5-30. Puede superar 100 si el cantante canta notas equivocadas.",
        example=12.4,
        ge=0,
        le=600
    )
    stabilityCents: float = Field(
        ...,
        description="Desviación tonal durante notas sostenidas (cents). Menor = más estable. Valores típicos: 3-20. Puede superar 50 si el cantante oscila entre notas.",
        example=8.9,
        ge=0,
        le=200
    )
    rangeMinMidi: float = Field(
        ...,
        description="Nota más grave del rango vocal (MIDI). Ejemplo: C3=48, C4=60",
        example=58.0,
        ge=20,
        le=108
    )
    rangeMaxMidi: float = Field(
        ...,
        description="Nota más aguda del rango vocal (MIDI). Ejemplo: C4=60, C5=72",
        example=80.0,
        ge=20,
        le=108
    )
    rangeSpanSemitones: float = Field(
        ...,
        description="Extensión vocal en semitonos. Típico: 12-36 semitonos (1-3 octavas)",
        example=22.0,
        ge=0,
        le=60
    )

    @root_validator(pre=True)
    def clamp_numeric_fields(cls, values):
        clamp_rules = {
            'meanRmsDb':          (-60.0,  0.0),
            'rmsConsistency':     (  0.0, None),
            'dynamicRangeDb':     (  0.0, 120.0),
            'durationSec':        (  0.0,  10.0),
            'attackLatencyMs':    (  0.0, 500.0),
            'precisionCents':     (  0.0, 600.0),
            'stabilityCents':     (  0.0, 200.0),
            'rangeMinMidi':       ( 20.0, 108.0),
            'rangeMaxMidi':       ( 20.0, 108.0),
            'rangeSpanSemitones': (  0.0,  60.0),
        }
        for field, (min_val, max_val) in clamp_rules.items():
            if field in values and isinstance(values[field], (int, float)):
                original = float(values[field])
                clamped = original
                if min_val is not None:
                    clamped = max(clamped, min_val)
                if max_val is not None:
                    clamped = min(clamped, max_val)
                if clamped != original:
                    logger.warning(f"Campo '{field}' fuera de rango ({original}), ajustado a {clamped}")
                values[field] = clamped
        return values

    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['F', 'M']:
            raise ValueError('El género debe ser "F" (femenino) o "M" (masculino)')
        return v

    @validator('rangeMaxMidi')
    def validate_range(cls, v, values):
        if 'rangeMinMidi' in values and v <= values['rangeMinMidi']:
            raise ValueError('rangeMaxMidi debe ser mayor que rangeMinMidi')
        return v

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "gender": "F",
                    "meanRmsDb": -25.5,
                    "rmsConsistency": 3.2,
                    "dynamicRangeDb": 70.0,
                    "durationSec": 2.8,
                    "attackLatencyMs": 75.0,
                    "precisionCents": 8.5,
                    "stabilityCents": 6.2,
                    "rangeMinMidi": 58.0,
                    "rangeMaxMidi": 80.0,
                    "rangeSpanSemitones": 22.0
                },
                {
                    "gender": "M",
                    "meanRmsDb": -45.0,
                    "rmsConsistency": 15.0,
                    "dynamicRangeDb": 25.0,
                    "durationSec": 1.2,
                    "attackLatencyMs": 150.0,
                    "precisionCents": 35.0,
                    "stabilityCents": 25.0,
                    "rangeMinMidi": 55.0,
                    "rangeMaxMidi": 65.0,
                    "rangeSpanSemitones": 10.0
                },
                {
                    "gender": "F",
                    "meanRmsDb": -30.0,
                    "rmsConsistency": 6.5,
                    "dynamicRangeDb": 50.0,
                    "durationSec": 2.2,
                    "attackLatencyMs": 95.0,
                    "precisionCents": 18.0,
                    "stabilityCents": 12.0,
                    "rangeMinMidi": 60.0,
                    "rangeMaxMidi": 78.0,
                    "rangeSpanSemitones": 18.0
                }
            ]
        }


class WeaknessDetection(BaseModel):
    """Resultado binario de detección de carencias por grupo"""
    weak_G1: int = Field(
        ...,
        description="1 = Carencia en Soporte Respiratorio y Control de Aire, 0 = Sin carencia",
        example=0,
        ge=0,
        le=1
    )
    weak_G2: int = Field(
        ...,
        description="1 = Carencia en Afinación y Oído Tonal, 0 = Sin carencia",
        example=0,
        ge=0,
        le=1
    )
    weak_G3: int = Field(
        ...,
        description="1 = Carencia en Estabilidad y Control, 0 = Sin carencia",
        example=0,
        ge=0,
        le=1
    )
    weak_G4: int = Field(
        ...,
        description="1 = Carencia en Potencia y Control Dinámico, 0 = Sin carencia",
        example=0,
        ge=0,
        le=1
    )
    weak_G5: int = Field(
        ...,
        description="1 = Carencia en Rango y Flexibilidad Vocal, 0 = Sin carencia",
        example=0,
        ge=0,
        le=1
    )


class PredictionResponse(BaseModel):
    """Respuesta completa de la predicción con carencias detectadas y niveles de confianza"""
    weaknesses_detected: List[str] = Field(
        ...,
        description="Lista de nombres de grupos con carencias detectadas",
        example=[]
    )
    total_weaknesses: int = Field(
        ...,
        description="Cantidad total de grupos con carencias detectadas (0-5)",
        example=0,
        ge=0,
        le=5
    )
    confidence_scores: Dict[str, float] = Field(
        ...,
        description="Probabilidad de carencia para cada grupo (0.0-1.0). Valores >0.5 indican carencia probable",
        example={
            "weak_G1": 0.0234,
            "weak_G2": 0.0156,
            "weak_G3": 0.0089,
            "weak_G4": 0.0312,
            "weak_G5": 0.0045
        }
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "summary": "Sin carencias detectadas",
                    "description": "Respuesta típica para un cantante con buena técnica",
                    "value": {
                        "weaknesses_detected": [],
                        "total_weaknesses": 0,
                        "confidence_scores": {
                            "weak_G1": 0.0234,
                            "weak_G2": 0.0156,
                            "weak_G3": 0.0089,
                            "weak_G4": 0.0312,
                            "weak_G5": 0.0045
                        }
                    }
                },
                {
                    "summary": "Múltiples carencias detectadas",
                    "description": "Respuesta cuando se detectan varios grupos con problemas",
                    "value": {
                        "weaknesses_detected": ["weak_G1", "weak_G4", "weak_G5"],
                        "total_weaknesses": 3,
                        "confidence_scores": {
                            "weak_G1": 0.8524,
                            "weak_G2": 0.1241,
                            "weak_G3": 0.2156,
                            "weak_G4": 0.7891,
                            "weak_G5": 0.9234
                        }
                    }
                }
            ]
        }


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud"""
    status: str = Field(..., description="Estado general del servicio", example="healthy")
    models_loaded: bool = Field(..., description="Indica si los 5 modelos XGBoost están cargados", example=True)
    scaler_loaded: bool = Field(..., description="Indica si el normalizador está cargado", example=True)
    encoder_loaded: bool = Field(..., description="Indica si el codificador de género está cargado", example=True)
    timestamp: str = Field(..., description="Timestamp de la verificación", example="2025-11-16T10:30:00")


class GroupMetric(BaseModel):
    """Métricas de evaluación por grupo para /predict"""
    score: float = Field(..., description="Probabilidad de carencia (0.0-1.0)", example=0.68)
    threshold: float = Field(..., description="Umbral de decisión para considerar carencia", example=0.45)
    is_weak: bool = Field(..., description="True si la probabilidad supera el umbral", example=True)
    missing_to_clear_pct: float = Field(
        ...,
        description="Porcentaje que falta para salir de zona de carencia (0-100). Si no hay carencia, es 0.",
        example=41.82
    )
    achievement_pct: float = Field(
        ...,
        description="Porcentaje de logro del grupo (100 cuando no hay carencia).",
        example=58.18
    )


class PredictionV2Response(BaseModel):
    """Respuesta extendida con porcentajes por grupo"""
    weaknesses_detected: List[str] = Field(..., description="Grupos detectados con carencia")
    total_weaknesses: int = Field(..., description="Total de grupos con carencia", ge=0, le=5)
    confidence_scores: Dict[str, float] = Field(..., description="Score/probabilidad por grupo")
    group_metrics: Dict[str, GroupMetric] = Field(..., description="Métricas avanzadas por grupo")

    class Config:
        json_schema_extra = {
            "example": {
                "weaknesses_detected": ["weak_G2", "weak_G4"],
                "total_weaknesses": 2,
                "confidence_scores": {
                    "weak_G1": 0.21,
                    "weak_G2": 0.68,
                    "weak_G3": 0.33,
                    "weak_G4": 0.74,
                    "weak_G5": 0.29
                },
                "group_metrics": {
                    "G1": {
                        "score": 0.21,
                        "threshold": 0.4,
                        "is_weak": False,
                        "missing_to_clear_pct": 0,
                        "achievement_pct": 100
                    },
                    "G2": {
                        "score": 0.68,
                        "threshold": 0.45,
                        "is_weak": True,
                        "missing_to_clear_pct": 41.82,
                        "achievement_pct": 58.18
                    },
                    "G3": {
                        "score": 0.33,
                        "threshold": 0.42,
                        "is_weak": False,
                        "missing_to_clear_pct": 0,
                        "achievement_pct": 100
                    },
                    "G4": {
                        "score": 0.74,
                        "threshold": 0.5,
                        "is_weak": True,
                        "missing_to_clear_pct": 48,
                        "achievement_pct": 52
                    },
                    "G5": {
                        "score": 0.29,
                        "threshold": 0.47,
                        "is_weak": False,
                        "missing_to_clear_pct": 0,
                        "achievement_pct": 100
                    }
                }
            }
        }


# ============================================================================
# ENDPOINTS
# ============================================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Salud"],
    summary="Verificar salud del servicio",
    description="Endpoint de health check para monitoreo. Verifica que todos los componentes estén cargados correctamente.",
    responses={
        200: {
            "description": "Servicio funcionando correctamente",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "models_loaded": True,
                        "scaler_loaded": True,
                        "encoder_loaded": True,
                        "timestamp": "2025-11-16T10:30:00"
                    }
                }
            }
        },
        503: {
            "description": "Servicio no disponible - Componentes faltantes"
        }
    }
)
async def health_check():
    """
    ## Health Check

    Verifica que el servicio esté funcionando correctamente:
    - ✅ Modelos XGBoost cargados (5 modelos)
    - ✅ StandardScaler cargado
    - ✅ Gender encoder cargado

    Este endpoint es útil para:
    - Monitoreo en producción
    - Load balancers
    - Verificación antes de hacer requests
    """
    is_healthy = (
        len(models) == 5 and
        scaler is not None and
        gender_encoder is not None
    )

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        models_loaded=len(models) == 5,
        scaler_loaded=scaler is not None,
        encoder_loaded=gender_encoder is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post(
    "/predict",
    response_model=PredictionV2Response,
    tags=["Predicción"],
    summary="Detectar carencias vocales",
    description="Incluye métricas por grupo para cuantificar avance/logro y brecha de mejora.",
    response_model_exclude_none=True
)
async def predict_weaknesses(metrics: VocalMetrics):
    """
    Versión avanzada de predicción.

    Además de detectar carencias, devuelve por cada grupo:
    - score de carencia
    - umbral aplicado
    - indicador booleano de carencia
    - porcentaje faltante para salir de zona de carencia
    - porcentaje de logro
    """
    try:
        logger.info(f"📊 Predicción solicitada para género: {metrics.gender}")

        if metrics.gender not in ['F', 'M']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El género debe ser 'F' o 'M'"
            )

        # Preparar features
        gender_encoded = gender_encoder.transform([metrics.gender])[0]
        features = np.array([[
            gender_encoded,
            metrics.meanRmsDb,
            metrics.rmsConsistency,
            metrics.dynamicRangeDb,
            metrics.durationSec,
            metrics.attackLatencyMs,
            metrics.precisionCents,
            metrics.stabilityCents,
            metrics.rangeMinMidi,
            metrics.rangeMaxMidi,
            metrics.rangeSpanSemitones
        ]])

        features_scaled = scaler.transform(features)

        confidence_scores = {}
        group_metrics = {}
        weaknesses_detected = []

        # Calcular scores y métricas por grupo usando umbrales configurables
        for idx, (group_name, model) in enumerate(models.items(), start=1):
            score = round(float(model.predict_proba(features_scaled)[0][1]), 4)
            threshold = GROUP_THRESHOLDS[group_name]

            forced_weak = (
                group_name in PHYSICAL_WEAKNESS_RULES
                and PHYSICAL_WEAKNESS_RULES[group_name](metrics)
            )
            is_weak = (score >= threshold) or forced_weak

            if is_weak:
                # Si el modelo no supera el umbral pero la regla física forzó debilidad,
                # usar 0.80 como score efectivo para los porcentajes (evidencia clara de carencia)
                effective_score = score if score >= threshold else 0.80
                missing_to_clear_pct = round(((effective_score - threshold) / (1 - threshold)) * 100, 2)
                achievement_pct = round(100 - missing_to_clear_pct, 2)
                weaknesses_detected.append(group_name)
            else:
                missing_to_clear_pct = 0.0
                achievement_pct = 100.0

            confidence_scores[group_name] = score
            group_metrics[f"G{idx}"] = GroupMetric(
                score=score,
                threshold=threshold,
                is_weak=is_weak,
                missing_to_clear_pct=missing_to_clear_pct,
                achievement_pct=achievement_pct
            )

        logger.info(f"✅ Predicción completada: {len(weaknesses_detected)} carencias detectadas")

        return PredictionV2Response(
            weaknesses_detected=weaknesses_detected,
            total_weaknesses=len(weaknesses_detected),
            confidence_scores=confidence_scores,
            group_metrics=group_metrics
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"❌ Error de validación: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar la predicción: {str(e)}"
        )


@app.post(
    "/batch-predict",
    tags=["Predicción"],
    summary="Detectar carencias en lote",
    responses={
        200: {
            "description": "Predicciones exitosas",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "total_predictions": 5,
                        "results": [
                            {
                                "weaknesses_detected": [],
                                "total_weaknesses": 0,
                                "confidence_scores": {
                                    "weak_G1": 0.0234,
                                    "weak_G2": 0.0156,
                                    "weak_G3": 0.0089,
                                    "weak_G4": 0.0312,
                                    "weak_G5": 0.0045
                                }
                            }
                        ]
                    }
                }
            }
        },
        500: {
            "description": "Error en el procesamiento batch"
        }
    }
)
async def batch_predict(metrics_list: List[VocalMetrics]):
    """
    Predice carencias para múltiples conjuntos de métricas.
    Útil si quieres evaluar varias vocales a la vez.
    """
    try:
        logger.info(f"📊 Batch prediction solicitada: {len(metrics_list)} muestras")
        results = []
        for i, metrics in enumerate(metrics_list, 1):
            logger.info(f"  Procesando muestra {i}/{len(metrics_list)}")
            result = await predict_weaknesses(metrics)
            results.append(result)

        logger.info(f"✅ Batch prediction completada: {len(results)} resultados")
        return {
            "success": True,
            "total_predictions": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"❌ Error en batch prediction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al procesar predicciones batch: {str(e)}"
        )


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Solo en desarrollo
        log_level="info"
    )

