from fastapi import Header, HTTPException, status
from typing import Optional
import os

API_KEYS = os.getenv("API_KEYS", "").split(",")

async def verify_api_key(api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Verify API key for authentication"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key