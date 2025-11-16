# UrSinger ML API

API REST para detección de carencias vocales usando modelos XGBoost.

## 🚀 Inicio Rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar el servicio

```bash
# En desarrollo (con auto-reload)
python main.py

# O con uvicorn directamente
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Acceder a la documentación

Una vez iniciado el servicio, abre:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 Endpoints

### `GET /`
Información del servicio

### `GET /health`
Verificar estado del servicio

### `POST /predict`
Detectar carencias vocales

**Request Body:**
```json
{
  "gender": "F",
  "meanRmsDb": -25.5,
  "rmsConsistency": 3.2,
  "dynamicRangeDb": 18.5,
  "durationSec": 2.8,
  "attackLatencyMs": 85.3,
  "precisionCents": 12.4,
  "stabilityCents": 8.9,
  "rangeMinMidi": 60.0,
  "rangeMaxMidi": 84.0,
  "rangeSpanSemitones": 24.0
}
```

**Response:**
```json
{
  "success": true,
  "weaknesses": {
    "weak_G1": 0,
    "weak_G2": 0,
    "weak_G3": 0,
    "weak_G4": 0,
    "weak_G5": 0
  },
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
```

## 🔗 Consumir desde NestJS

### Ejemplo de servicio en NestJS

```typescript
// src/ml/ml.service.ts
import { Injectable, HttpException, HttpStatus } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';

interface VocalMetrics {
  gender: string;
  meanRmsDb: number;
  rmsConsistency: number;
  dynamicRangeDb: number;
  durationSec: number;
  attackLatencyMs: number;
  precisionCents: number;
  stabilityCents: number;
  rangeMinMidi: number;
  rangeMaxMidi: number;
  rangeSpanSemitones: number;
}

interface WeaknessDetection {
  weak_G1: number;
  weak_G2: number;
  weak_G3: number;
  weak_G4: number;
  weak_G5: number;
}

interface PredictionResponse {
  success: boolean;
  weaknesses: WeaknessDetection;
  weaknesses_detected: string[];
  total_weaknesses: number;
  confidence_scores: Record<string, number>;
}

@Injectable()
export class MlService {
  private readonly ML_API_URL = process.env.ML_API_URL || 'http://localhost:8000';

  constructor(private readonly httpService: HttpService) {}

  async detectWeaknesses(metrics: VocalMetrics): Promise<PredictionResponse> {
    try {
      const response = await firstValueFrom(
        this.httpService.post<PredictionResponse>(
          `${this.ML_API_URL}/predict`,
          metrics
        )
      );
      return response.data;
    } catch (error) {
      throw new HttpException(
        'Error al consultar el servicio ML',
        HttpStatus.INTERNAL_SERVER_ERROR
      );
    }
  }

  async generateLearningPath(weaknesses: string[]): Promise<any> {
    // AQUÍ generas la ruta de aprendizaje según las carencias
    const learningPaths = {
      weak_G1: {
        name: 'Soporte Respiratorio',
        exercises: [
          { id: 1, name: 'Respiración diafragmática', duration: 10 },
          { id: 2, name: 'Sostener notas largas', duration: 15 },
        ]
      },
      weak_G2: {
        name: 'Afinación y Oído Tonal',
        exercises: [
          { id: 3, name: 'Escalas con metrónomo', duration: 20 },
          { id: 4, name: 'Intervalos ascendentes', duration: 15 },
        ]
      },
      // ... más grupos
    };

    return weaknesses.map(w => learningPaths[w] || null).filter(Boolean);
  }
}
```

## 🌐 En Producción

### Opción 1: Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Opción 2: Servidor Linux con systemd
```ini
[Unit]
Description=UrSinger ML API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/ursinger-ml
ExecStart=/usr/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## 🔒 Seguridad

- Usar HTTPS en producción
- Configurar CORS correctamente
- Implementar rate limiting
- Agregar autenticación (JWT)
- Validar todas las entradas

## 📊 Monitoreo

- Logs en `/var/log/ursinger-ml/`
- Métricas con Prometheus
- Health check: `GET /health`

