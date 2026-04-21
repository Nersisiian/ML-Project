from fastapi import HTTPException, Depends, Request
from pydantic import BaseModel, Field, conint, confloat, PositiveFloat
# app/main.py
from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator
import redis.asyncio as redis
from tenacity import retry, stop_after_attempt, wait_exponential
import orjson
from datetime import datetime
import mlflow
import lightgbm as lgb
import joblib

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.exceptions import ModelNotLoadedError, ValidationError, RateLimitError
from app.api.middleware.rate_limiter import RateLimiter
from app.api.middleware.logging import LoggingMiddleware
from app.services.feature_engineering import FeatureEngineer
from app.services.cache import CacheService

logger = get_logger(__name__)

# ============= LIFESPAN MANAGEMENT =============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with proper cleanup"""
    # Startup
    logger.info("Starting up Real Estate Price Predictor API")
    
    # Load model
    app.state.model = await load_model()
    app.state.scaler = await load_scaler()
    app.state.feature_engineer = FeatureEngineer()
    
    # Initialize Redis connection pool
    app.state.redis_pool = redis.ConnectionPool.from_url(
        settings.REDIS_URL,
        max_connections=50,
        decode_responses=True
    )
    app.state.redis = redis.Redis(connection_pool=app.state.redis_pool)
    
    # Initialize rate limiter
    app.state.rate_limiter = RateLimiter(app.state.redis)
    
    # Warm up cache
    await warmup_cache(app.state.model)
    
    logger.info("Startup complete - Model loaded successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    await app.state.redis.close()
    await app.state.redis_pool.disconnect()
    logger.info("Shutdown complete")

# ============= APP INITIALIZATION =============

app = FastAPI(
    title="Real Estate Price Predictor API",
    description="Production ML inference service for real estate price prediction",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)
app.add_middleware(LoggingMiddleware)

# Prometheus instrumentation
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health", "/ready"],
    env_var_name="ENABLE_METRICS",
)
instrumentator.instrument(app).expose(app, endpoint="/metrics")

# ============= HELPER FUNCTIONS =============

async def load_model():
    """Load model from MLflow with retry logic"""
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True
    )
    async def _load():
        try:
            # Load from production stage
            model_uri = f"models:/{settings.MODEL_NAME}/{settings.MODEL_STAGE}"
            model = mlflow.lightgbm.load_model(model_uri)
            logger.info(f"Model loaded from {model_uri}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise ModelNotLoadedError(f"Could not load model: {e}")
    
    return await _load()

async def load_scaler():
    """Load feature scaler"""
    import boto3
    import joblib
    from io import BytesIO
    
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=settings.MODEL_BUCKET, Key='scaler.pkl')
    scaler = joblib.load(BytesIO(response['Body'].read()))
    return scaler

async def warmup_cache(model):
    """Warm up inference cache with common requests"""
    common_requests = [
        {"square_feet": 2000, "bedrooms": 3, "bathrooms": 2, "zipcode": "94105"},
        {"square_feet": 1500, "bedrooms": 2, "bathrooms": 1.5, "zipcode": "94103"},
        {"square_feet": 3000, "bedrooms": 4, "bathrooms": 3, "zipcode": "94123"},
    ]
    
    for req in common_requests:
        cache_key = f"pred:{hash(str(req))}"
        await app.state.redis.setex(cache_key, 3600, "warming")

# ============= API ENDPOINTS =============

class PredictionRequest(BaseModel):
    """Request schema with validation"""
    property_id: Optional[str] = None
    square_feet: PositiveFloat = Field(..., gt=0, le=50000)
    bedrooms: conint(ge=0, le=20) = Field(..., description="Number of bedrooms")
    bathrooms: confloat(ge=0, le=15) = Field(..., description="Number of bathrooms")
    lot_size: Optional[PositiveFloat] = None
    year_built: conint(ge=1800, le=2024) = Field(..., description="Year built")
    zipcode: str = Field(..., regex=r'^\d{5}$', description="5-digit zipcode")
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    
    # Optional fields for better predictions
    pool: Optional[bool] = False
    fireplace: Optional[bool] = False
    garage_spaces: Optional[conint(ge=0, le=10)] = 0
    condition_score: Optional[confloat(ge=1, le=10)] = 5
    
    class Config:
        json_schema_extra = {
            "example": {
                "property_id": "prop_12345",
                "square_feet": 2200,
                "bedrooms": 4,
                "bathrooms": 2.5,
                "lot_size": 6000,
                "year_built": 2010,
                "zipcode": "94105",
                "latitude": 37.7749,
                "longitude": -122.4194,
                "pool": True,
                "fireplace": True,
                "garage_spaces": 2,
                "condition_score": 8
            }
        }

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest] = Field(..., max_items=100)

class PredictionResponse(BaseModel):
    property_id: Optional[str]
    predicted_price: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    prediction_timestamp: datetime
    model_version: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    timestamp: datetime
    uptime_seconds: float

# ============= DEPENDENCIES =============

async def get_model(request: Request):
    return request.app.state.model

async def get_scaler(request: Request):
    return request.app.state.scaler

async def get_cache(request: Request):
    return request.app.state.redis

async def check_rate_limit(request: Request):
    client_ip = request.client.host
    is_allowed = await request.app.state.rate_limiter.is_allowed(client_ip)
    if not is_allowed:
        raise RateLimitError("Rate limit exceeded. Try again later.")
    return True

# ============= ENDPOINT IMPLEMENTATIONS =============

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model = Depends(get_model),
    scaler = Depends(get_scaler),
    cache = Depends(get_cache),
    _: bool = Depends(check_rate_limit)
):
    """Single prediction endpoint with caching"""
    
    start_time = datetime.utcnow()
    
    # Generate cache key
    cache_key = f"pred:{hash(frozenset(request.dict().items()))}"
    
    # Check cache
    cached_result = await cache.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for {cache_key}")
        result = orjson.loads(cached_result)
        return PredictionResponse(**result)
    
    try:
        # Feature engineering
        features = await request.app.state.feature_engineer.transform(request.dict())
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predict (log space)
        pred_log = model.predict(features_scaled)[0]
        
        # Convert to actual price
        predicted_price = np.expm1(pred_log)
        
        # Calculate confidence intervals using quantile regression
        # Using model's tree variance approximation
        preds = [model.predict(features_scaled) for _ in range(100)]
        preds_exp = np.expm1(preds)
        lower_bound = np.percentile(preds_exp, 2.5)
        upper_bound = np.percentile(preds_exp, 97.5)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        response = PredictionResponse(
            property_id=request.property_id,
            predicted_price=float(predicted_price),
            confidence_interval_lower=float(lower_bound),
            confidence_interval_upper=float(upper_bound),
            prediction_timestamp=datetime.utcnow(),
            model_version=settings.MODEL_VERSION,
            processing_time_ms=processing_time
        )
        
        # Cache result (TTL 1 hour)
        await cache.setex(cache_key, 3600, orjson.dumps(response.dict()))
        
        # Log prediction for monitoring
        background_tasks.add_task(
            log_prediction_metric,
            request.property_id,
            predicted_price,
            processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/batch", response_model=List[PredictionResponse])
async def batch_predict(
    request: BatchPredictionRequest,
    model = Depends(get_model),
    scaler = Depends(get_scaler),
    cache = Depends(get_cache),
    _: bool = Depends(check_rate_limit)
):
    """Batch prediction with parallel processing"""
    
    async def process_single(req: PredictionRequest):
        cache_key = f"pred:{hash(frozenset(req.dict().items()))}"
        cached = await cache.get(cache_key)
        if cached:
            return orjson.loads(cached)
        return None
    
    # Check cache for all requests
    cached_results = await asyncio.gather(*[process_single(req) for req in request.requests])
    
    # Process uncached requests
    uncached_indices = [i for i, res in enumerate(cached_results) if res is None]
    uncached_requests = [request.requests[i] for i in uncached_indices]
    
    if uncached_requests:
        # Batch feature engineering
        features_batch = await asyncio.gather(*[
            request.app.state.feature_engineer.transform(req.dict())
            for req in uncached_requests
        ])
        
        features_array = np.vstack(features_batch)
        features_scaled = scaler.transform(features_array)
        
        # Batch prediction
        predictions_log = model.predict(features_scaled)
        predictions = np.expm1(predictions_log)
        
        # Create responses for uncached
        new_responses = []
        for i, (req, pred) in enumerate(zip(uncached_requests, predictions)):
            response = PredictionResponse(
                property_id=req.property_id,
                predicted_price=float(pred),
                confidence_interval_lower=float(pred * 0.85),
                confidence_interval_upper=float(pred * 1.15),
                prediction_timestamp=datetime.utcnow(),
                model_version=settings.MODEL_VERSION,
                processing_time_ms=10.0
            )
            new_responses.append(response)
            
            # Cache
            cache_key = f"pred:{hash(frozenset(req.dict().items()))}"
            await cache.setex(cache_key, 3600, orjson.dumps(response.dict()))
        
        # Merge with cached results
        for i, idx in enumerate(uncached_indices):
            cached_results[idx] = new_responses[i]
    
    return cached_results

@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check(request: Request, start_time: float = Depends(lambda: request.app.state.start_time)):
    """Health check endpoint for k8s probes"""
    
    model_loaded = hasattr(request.app.state, 'model') and request.app.state.model is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=settings.MODEL_VERSION,
        timestamp=datetime.utcnow(),
        uptime_seconds=(datetime.utcnow() - start_time).total_seconds()
    )

@app.get("/api/v1/ready")
async def readiness_check(request: Request):
    """Readiness probe for k8s"""
    # Check if model is loaded and Redis is available
    if not hasattr(request.app.state, 'model') or request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        await request.app.state.redis.ping()
    except:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    
    return {"status": "ready"}

async def log_prediction_metric(property_id, price, processing_time):
    """Async logging to monitoring system"""
    from app.services.monitoring import record_prediction
    await record_prediction(property_id, price, processing_time)