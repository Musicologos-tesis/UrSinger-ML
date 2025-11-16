"""
Script de prueba para verificar que la API ML funciona correctamente
"""

import requests
import json

# URL de la API
API_URL = "http://localhost:8000"

def test_health():
    """Verifica que el servicio esté corriendo"""
    print("🔍 Verificando estado del servicio...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_root():
    """Verifica el endpoint raíz"""
    print("🔍 Consultando endpoint raíz...")
    response = requests.get(f"{API_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_predict_no_weaknesses():
    """Prueba con métricas de un cantante profesional (sin carencias)"""
    print("🎤 Test 1: Cantante profesional (sin carencias esperadas)")

    metrics = {
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
    }

    response = requests.post(f"{API_URL}/predict", json=metrics)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    print(f"✅ Carencias detectadas: {result['total_weaknesses']}")
    print(f"✅ Grupos con carencias: {result['weaknesses_detected']}\n")

def test_predict_with_weaknesses():
    """Prueba con métricas de un cantante con carencias evidentes"""
    print("🎤 Test 2: Cantante con carencias (múltiples carencias esperadas)")

    # Métricas con problemas evidentes
    metrics = {
        "gender": "M",
        "meanRmsDb": -45.0,  # Muy bajo (problema de potencia G4)
        "rmsConsistency": 15.0,  # Alta inconsistencia (problema G1, G3, G4)
        "dynamicRangeDb": 25.0,  # Rango dinámico limitado (problema G4)
        "durationSec": 1.2,  # Duración corta (problema G1)
        "attackLatencyMs": 150.0,  # Latencia alta (problema G3)
        "precisionCents": 35.0,  # Pobre afinación (problema G2)
        "stabilityCents": 25.0,  # Inestable (problema G2, G3)
        "rangeMinMidi": 55.0,
        "rangeMaxMidi": 65.0,
        "rangeSpanSemitones": 10.0  # Rango limitado (problema G5)
    }

    response = requests.post(f"{API_URL}/predict", json=metrics)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    print(f"⚠️  Carencias detectadas: {result['total_weaknesses']}")
    print(f"⚠️  Grupos con carencias: {result['weaknesses_detected']}")

    # Mostrar probabilidades
    print("\n📊 Probabilidades de carencia:")
    for group, prob in result['confidence_scores'].items():
        print(f"  {group}: {prob*100:.1f}%")
    print()

def test_batch_predict():
    """Prueba predicción en batch (múltiples métricas)"""
    print("🎤 Test 3: Predicción batch (5 vocales)")

    metrics_list = [
        {
            "gender": "F",
            "meanRmsDb": -26.0,
            "rmsConsistency": 3.5,
            "dynamicRangeDb": 68.0,
            "durationSec": 2.7,
            "attackLatencyMs": 80.0,
            "precisionCents": 9.0,
            "stabilityCents": 7.0,
            "rangeMinMidi": 58.0,
            "rangeMaxMidi": 79.0,
            "rangeSpanSemitones": 21.0
        },
        # ... puedes agregar más
    ]

    response = requests.post(f"{API_URL}/batch-predict", json=metrics_list)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"✅ Total de predicciones: {result['total_predictions']}\n")

def main():
    print("="*70)
    print("  PRUEBAS DE LA API ML - UrSinger")
    print("="*70)
    print()

    try:
        test_health()
        test_root()
        test_predict_no_weaknesses()
        test_predict_with_weaknesses()
        # test_batch_predict()  # Descomentar si quieres probarlo

        print("="*70)
        print("  ✅ TODAS LAS PRUEBAS COMPLETADAS")
        print("="*70)

    except requests.exceptions.ConnectionError:
        print("❌ ERROR: No se pudo conectar a la API")
        print("   Asegúrate de que el servicio esté corriendo en http://localhost:8000")
        print("   Ejecuta: python api/main.py")
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    main()

