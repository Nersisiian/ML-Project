from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge
import psutil
import platform
from datetime import datetime

router = APIRouter()

# Custom metrics
prediction_counter = Counter(
    'predictions_total',
    'Total number of predictions',
    ['model_version', 'status']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
)

model_accuracy = Gauge(
    'model_accuracy',
    'Current model accuracy metrics',
    ['metric']
)

cache_hits = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

cache_misses = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

data_drift_score = Gauge(
    'data_drift_score',
    'Data drift detection score',
    ['feature']
)

active_requests = Gauge(
    'active_requests',
    'Currently active requests'
)

request_duration = Histogram(
    'request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

# System metrics
system_cpu_usage = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage'
)

system_memory_usage = Gauge(
    'system_memory_usage_bytes',
    'System memory usage in bytes'
)

system_disk_usage = Gauge(
    'system_disk_usage_percent',
    'System disk usage percentage'
)

@router.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics"""
    
    # Update system metrics
    system_cpu_usage.set(psutil.cpu_percent(interval=1))
    system_memory_usage.set(psutil.virtual_memory().used)
    system_disk_usage.set(psutil.disk_usage('/').percent)
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@router.get("/metrics/system")
async def get_system_metrics():
    """Get system metrics as JSON"""
    
    return {
        "cpu": {
            "usage_percent": psutil.cpu_percent(interval=1),
            "cores": psutil.cpu_count(),
            "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None
        },
        "memory": {
            "total_bytes": psutil.virtual_memory().total,
            "available_bytes": psutil.virtual_memory().available,
            "used_percent": psutil.virtual_memory().percent
        },
        "disk": {
            "total_bytes": psutil.disk_usage('/').total,
            "used_bytes": psutil.disk_usage('/').used,
            "used_percent": psutil.disk_usage('/').percent
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version()
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/metrics/model")
async def get_model_metrics():
    """Get model performance metrics"""
    
    return {
        "model_version": "2.0.0",
        "metrics": {
            "mae": 25000,
            "rmse": 35000,
            "r2": 0.89,
            "mape": 12.5
        },
        "predictions_total": prediction_counter._value.get(),
        "cache_hit_ratio": calculate_cache_hit_ratio(),
        "timestamp": datetime.utcnow().isoformat()
    }

def calculate_cache_hit_ratio() -> float:
    """Calculate cache hit ratio"""
    hits = cache_hits._value.get()
    misses = cache_misses._value.get()
    total = hits + misses
    return hits / total if total > 0 else 0.0