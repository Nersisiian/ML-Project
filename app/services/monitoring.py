# app/services/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import asyncio
from typing import Dict, Any
import numpy as np

# Prometheus metrics
PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total number of predictions',
    ['model_version', 'status']
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
)

MODEL_SCORE = Gauge(
    'model_score',
    'Current model performance metrics',
    ['metric']
)

DATA_DRIFT_SCORE = Gauge(
    'data_drift_score',
    'Data drift detection score',
    ['feature']
)

CACHE_HIT_RATIO = Gauge(
    'cache_hit_ratio',
    'Redis cache hit ratio'
)

async def record_prediction(property_id: str, price: float, latency_ms: float):
    """Record prediction metrics"""
    PREDICTION_COUNT.labels(model_version='v2', status='success').inc()
    PREDICTION_LATENCY.observe(latency_ms / 1000)
    
    # Log to structured logging
    logger.info(
        "Prediction recorded",
        extra={
            "property_id": property_id,
            "predicted_price": price,
            "latency_ms": latency_ms
        }
    )

async def calculate_drift(reference_dist: np.ndarray, current_dist: np.ndarray):
    """Calculate population stability index (PSI) for drift detection"""
    psi = np.sum((current_dist - reference_dist) * np.log(current_dist / reference_dist))
    
    if psi > 0.2:
        logger.warning(f"Significant drift detected: PSI={psi}")
        # Trigger alert
        await send_alert(f"Data drift detected! PSI={psi}")
    
    return psi