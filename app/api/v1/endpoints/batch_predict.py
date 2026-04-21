from fastapi import APIRouter, Depends, HTTPException, Request
from typing import List
import asyncio
import numpy as np
from datetime import datetime

from app.api.v1.endpoints.predict import PredictionRequest, PredictionResponse

router = APIRouter()

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest] = Field(..., max_items=100)

@router.post("/predict/batch", response_model=List[PredictionResponse])
async def batch_predict(
    request: BatchPredictionRequest,
    req: Request,
    api_key: str = Depends(verify_api_key)
):
    """Batch prediction endpoint"""
    
    # Process in parallel with semaphore
    semaphore = asyncio.Semaphore(10)
    
    async def process_with_limit(pred_request):
        async with semaphore:
            # Reuse single prediction logic
            from app.api.v1.endpoints.predict import predict as single_predict
            return await single_predict(pred_request, None, req, api_key)
    
    tasks = [process_with_limit(r) for r in request.requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle errors
    responses = []
    for result in results:
        if isinstance(result, Exception):
            raise HTTPException(status_code=500, detail=str(result))
        responses.append(result)
    
    return responses