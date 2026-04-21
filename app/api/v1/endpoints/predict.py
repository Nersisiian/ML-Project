from fastapi import HTTPException, Depends, Request
from pydantic import BaseModel, Field, conint, confloat, PositiveFloat
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field, confloat, conint
from typing import Optional, List
from datetime import datetime
import numpy as np
import orjson
from prometheus_client import Counter, Histogram

from app.core.exceptions import ValidationError
from app.services.inference import InferenceService
from app.services.cache import CacheService
from app.dependencies.rate_limiter import RateLimiter
from app.dependencies.auth import verify_api_key

router = APIRouter()

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total predictions', ['status'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

class PredictionRequest(BaseModel):
    """Prediction request schema"""
    property_id: Optional[str] = None
    square_feet: confloat(gt=0, le=50000) = Field(..., description="Square footage")
    bedrooms: conint(ge=0, le=20) = Field(..., description="Number of bedrooms")
    bathrooms: confloat(ge=0, le=15) = Field(..., description="Number of bathrooms")
    lot_size: Optional[confloat(gt=0)] = None
    year_built: conint(ge=1800, le=2024) = Field(..., description="Year built")
    zipcode: str = Field(..., regex=r'^\d{5}$')
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    pool: bool = False
    fireplace: bool = False
    garage_spaces: conint(ge=0, le=10) = 0
    condition_score: confloat(ge=1, le=10) = 5
    
    class Config:
        json_schema_extra = {
            "example": {
                "property_id": "prop_123",
                "square_feet": 2200,
                "bedrooms": 4,
                "bathrooms": 2.5,
                "year_built": 2010,
                "zipcode": "94105",
                "latitude": 37.7749,
                "longitude": -122.4194
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response schema"""
    property_id: Optional[str]
    predicted_price: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    prediction_timestamp: datetime
    model_version: str
    processing_time_ms: float

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    req: Request,
    api_key: str = Depends(verify_api_key),
    rate_limiter: RateLimiter = Depends(lambda: req.app.state.rate_limiter)
):
    """Single prediction endpoint"""
    
    start_time = datetime.utcnow()
    
    # Check rate limit
    allowed, headers = await rate_limiter.is_allowed(req.client.host)
    if not allowed:
        prediction_counter.labels(status='rate_limited').inc()
        raise HTTPException(status_code=429, detail="Rate limit exceeded", headers=headers)
    
    # Check cache
    cache_key = f"pred:{hash(frozenset(request.dict().items()))}"
    cached = await req.app.state.cache.get(cache_key)
    
    if cached:
        prediction_counter.labels(status='cache_hit').inc()
        return PredictionResponse(**orjson.loads(cached))
    
    try:
        # Get inference service
        inference_service = InferenceService(
            model=req.app.state.model,
            scaler=req.app.state.scaler if hasattr(req.app.state, 'scaler') else None
        )
        
        # Make prediction
        features = inference_service.preprocess(request.dict())
        prediction = inference_service.predict(features)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        response = PredictionResponse(
            property_id=request.property_id,
            predicted_price=float(prediction['price']),
            confidence_interval_lower=float(prediction['ci_lower']),
            confidence_interval_upper=float(prediction['ci_upper']),
            prediction_timestamp=datetime.utcnow(),
            model_version=req.app.state.model_version if hasattr(req.app.state, 'model_version') else "unknown",
            processing_time_ms=processing_time
        )
        
        # Cache response
        await req.app.state.cache.set(cache_key, orjson.dumps(response.dict()), ttl=3600)
        
        prediction_counter.labels(status='success').inc()
        prediction_latency.observe(processing_time / 1000)
        
        return response
        
    except Exception as e:
        prediction_counter.labels(status='error').inc()
        raise HTTPException(status_code=500, detail=str(e))