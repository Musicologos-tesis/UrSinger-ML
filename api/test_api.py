"""Script de pruebas para verificar que la API ML responde correctamente."""

import json
import os
from typing import Dict, Any, List

import pandas as pd
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
DATASET_PATH = os.getenv("DATASET_PATH", "data/training_data_augmented.csv")
SWEEP_SAMPLES = int(os.getenv("SWEEP_SAMPLES", "50"))
TIMEOUT_SEC = 10


def _request(method: str, path: str, payload: Dict[str, Any] = None):
    url = f"{API_URL}{path}"
    response = requests.request(method, url, json=payload, timeout=TIMEOUT_SEC)
    try:
        body = response.json()
    except Exception:
        body = response.text
    return response, body


def _validate_prediction(result: Dict[str, Any]):
    required_keys = {"weaknesses_detected", "total_weaknesses", "confidence_scores", "group_metrics"}
    if not required_keys.issubset(result.keys()):
        missing = required_keys - set(result.keys())
        raise AssertionError(f"Missing keys in response: {missing}")

    total = result["total_weaknesses"]
    if not isinstance(total, int) or total < 0 or total > 5:
        raise AssertionError("total_weaknesses must be an int in [0, 5]")

    scores = result["confidence_scores"]
    if not isinstance(scores, dict) or len(scores) != 5:
        raise AssertionError("confidence_scores must contain 5 groups")
    for value in scores.values():
        if not (0.0 <= float(value) <= 1.0):
            raise AssertionError("confidence_scores values must be in [0, 1]")

    group_metrics = result["group_metrics"]
    if not isinstance(group_metrics, dict) or len(group_metrics) != 5:
        raise AssertionError("group_metrics must contain 5 groups")

    for metrics in group_metrics.values():
        for key in ["score", "threshold", "is_weak", "missing_to_clear_pct", "achievement_pct"]:
            if key not in metrics:
                raise AssertionError(f"group_metrics missing key: {key}")
        if not (0.0 <= float(metrics["score"]) <= 1.0):
            raise AssertionError("score must be in [0, 1]")
        if not (0.0 <= float(metrics["threshold"]) <= 1.0):
            raise AssertionError("threshold must be in [0, 1]")
        if not (0.0 <= float(metrics["missing_to_clear_pct"]) <= 100.0):
            raise AssertionError("missing_to_clear_pct must be in [0, 100]")
        if not (0.0 <= float(metrics["achievement_pct"]) <= 100.0):
            raise AssertionError("achievement_pct must be in [0, 100]")


def test_health():
    print("[health] Checking service status...")
    response, body = _request("GET", "/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(body, indent=2))
    if response.status_code != 200:
        raise AssertionError("Health check failed")
    print()


def test_root():
    print("[root] Checking root endpoint...")
    response, body = _request("GET", "/")
    print(f"Status: {response.status_code}")
    if response.status_code == 404:
        print("Root endpoint not defined (expected).")
    else:
        print(json.dumps(body, indent=2))
    print()


def _post_predict(label: str, metrics: Dict[str, Any]):
    print(f"[predict] {label}")
    response, body = _request("POST", "/predict", metrics)
    print(f"Status: {response.status_code}")
    if response.status_code != 200:
        raise AssertionError(f"Predict failed: {body}")
    _validate_prediction(body)
    print(json.dumps(body, indent=2))
    print(f"Total weaknesses: {body['total_weaknesses']}")
    print(f"Weak groups: {body['weaknesses_detected']}\n")
    return body


def test_predict_no_weaknesses():
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
    _post_predict("Professional singer (expected no weaknesses)", metrics)


def _build_multi_candidate(df: pd.DataFrame) -> Dict[str, Any]:
    worst_high = {
        "rmsConsistency",
        "attackLatencyMs",
        "precisionCents",
        "stabilityCents",
        "rangeMinMidi"
    }
    worst_low = {
        "meanRmsDb",
        "dynamicRangeDb",
        "durationSec",
        "rangeMaxMidi",
        "rangeSpanSemitones"
    }

    candidate = {"gender": "M"}
    if "gender" in df.columns:
        candidate["gender"] = df["gender"].mode().iloc[0]

    for col in worst_high:
        if col in df.columns:
            candidate[col] = float(df[col].max())
    for col in worst_low:
        if col in df.columns:
            candidate[col] = float(df[col].min())

    range_min = candidate.get("rangeMinMidi")
    range_max = candidate.get("rangeMaxMidi")
    if range_min is not None and range_max is not None:
        if range_max <= range_min:
            range_max = float(range_min) + 1.0
            candidate["rangeMaxMidi"] = range_max
        candidate["rangeSpanSemitones"] = max(1.0, range_max - range_min)

    return candidate


def test_predict_multi_candidate():
    if not os.path.exists(DATASET_PATH):
        print("[multi] Dataset not found, skipping multi-weakness candidate")
        return

    df = pd.read_csv(DATASET_PATH)
    metrics = _build_multi_candidate(df)
    _post_predict("Multi-weakness candidate (extreme metrics)", metrics)


def test_dataset_sweep():
    if not os.path.exists(DATASET_PATH):
        print("[sweep] Dataset not found, skipping sweep")
        return

    df = pd.read_csv(DATASET_PATH)
    sample_df = df.sample(n=min(SWEEP_SAMPLES, len(df)), random_state=42)

    totals: List[int] = []
    multi_hits = 0
    for _, row in sample_df.iterrows():
        metrics = {
            "gender": row.get("gender", "F"),
            "meanRmsDb": float(row["meanRmsDb"]),
            "rmsConsistency": float(row["rmsConsistency"]),
            "dynamicRangeDb": float(row["dynamicRangeDb"]),
            "durationSec": float(row["durationSec"]),
            "attackLatencyMs": float(row["attackLatencyMs"]),
            "precisionCents": float(row["precisionCents"]),
            "stabilityCents": float(row["stabilityCents"]),
            "rangeMinMidi": float(row["rangeMinMidi"]),
            "rangeMaxMidi": float(row["rangeMaxMidi"]),
            "rangeSpanSemitones": float(row["rangeSpanSemitones"])
        }
        response, body = _request("POST", "/predict", metrics)
        if response.status_code != 200:
            raise AssertionError(f"Sweep predict failed: {body}")
        _validate_prediction(body)
        totals.append(body["total_weaknesses"])
        if body["total_weaknesses"] >= 2:
            multi_hits += 1

    if totals:
        min_total = min(totals)
        max_total = max(totals)
        avg_total = sum(totals) / len(totals)
        print("[sweep] Summary over dataset sample")
        print(f"Samples: {len(totals)}")
        print(f"Total weaknesses -> min: {min_total}, avg: {avg_total:.2f}, max: {max_total}")
        print(f"Samples with >=2 weaknesses: {multi_hits}")
    print()


def main():
    print("=" * 70)
    print("  API TESTS - UrSinger")
    print("=" * 70)
    print()

    try:
        test_health()
        test_root()
        test_predict_no_weaknesses()
        test_predict_multi_candidate()
        test_dataset_sweep()

        print("=" * 70)
        print("  ALL TESTS COMPLETED")
        print("=" * 70)

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API")
        print(f"Make sure the service is running at {API_URL}")
        print("Run: py api/main.py")
    except Exception as exc:
        print(f"ERROR: {exc}")


if __name__ == "__main__":
    main()

