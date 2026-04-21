from fastapi import APIRouter, Request
from datetime import datetime
from pydantic import BaseModel

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    timestamp: datetime
    uptime_seconds: float

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint"""
    
    uptime = (datetime.utcnow() - request.app.state.start_time).total_seconds()
    
    return HealthResponse(
        status="healthy" if request.app.state.model else "degraded",
        model_loaded=request.app.state.model is not None,
        model_version=getattr(request.app.state, 'model_version', 'unknown'),
        timestamp=datetime.utcnow(),
        uptime_seconds=uptime
    )

@router.get("/ready")
async def readiness_check(request: Request):
    """Readiness probe"""
    
    if not request.app.state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check Redis
    try:
        await request.app.state.redis.ping()
    except:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    
    return {"status": "ready"}